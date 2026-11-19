"""Independently review batch32 pilot predictions, arithmetic and saved evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import runtime
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=runtime.PROJECT_ROOT / '05_experiments' / 'batch32_v2_pilot')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not root.is_relative_to(runtime.PROJECT_ROOT / '05_experiments'):
        raise ValueError('Use a run inside project/05_experiments')
    prior = json.loads((root / 'integrity.json').read_text())
    for name, digest in prior.items():
        assert sha(root / name) == digest, f'Integrity failure: {name}'
    rows = list(csv.DictReader((root / 'runs.csv').open()))
    results = json.loads((root / 'results_explained.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    environment = json.loads((root / 'environment.json').read_text())
    checks = []

    def check(name, condition, detail=''):
        checks.append({'check': name, 'passed': bool(condition), 'detail': str(detail)})
        if not condition:
            raise AssertionError(f'{name}: {detail}')

    check('12 unique runs', len(rows) == 12 and len({r['run_id'] for r in rows}) == 12)
    expected_indices = np.load(runtime.PROJECT_ROOT / '03_data' / 'manifests' / 'val_indices.npy')
    probabilities = {}
    for row in rows:
        rid = row['run_id']
        with np.load(root / 'predictions' / f'{rid}_validation.npz') as data:
            y, p, indices = data['labels'], data['probabilities'], data['indices']
        check(rid + ': validation indices', np.array_equal(indices, expected_indices))
        check(rid + ': finite probabilities in [0,1]', len(p) == 524 and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)))
        pos, neg = p[y == 1], p[y == 0]
        # Independent positive/negative pair counting, not sklearn rank-sum AUROC.
        auc = float(np.mean((pos[:, None] > neg[None, :]).astype(float) + .5 * (pos[:, None] == neg[None, :])))
        tp = np.sum((p >= .5) & (y == 1))
        fp = np.sum((p >= .5) & (y == 0))
        fn = np.sum((p < .5) & (y == 1))
        f1 = float(2 * tp / (2 * tp + fp + fn))
        check(rid + ': independent AUROC', abs(auc - float(row['validation_auroc'])) < 1e-12, auc)
        check(rid + ': independent F1', abs(f1 - float(row['validation_f1'])) < 1e-12, f1)
        check(rid + ': optimizer update count', int(row['total_optimizer_steps']) == 3 * int(np.ceil(4698 / int(row['batch']))))
        check(rid + ': timing and memory positive', float(row['adaptation_seconds']) > 0 and float(row['peak_allocated_mib']) > 0)
        history = json.loads((root / 'history' / f'{rid}.json').read_text())
        check(rid + ': checkpoint hash', sha(root / 'checkpoints' / f'{rid}.pt') == history['checkpoint_sha256'])
        probabilities[(int(row['seed']), int(row['batch']), int(row['session']))] = (y, p, history)
    for seed in [42, 43, 44]:
        for batch in [16, 32]:
            a, b = [probabilities[(seed, batch, s)] for s in [1, 2]]
            check(f'{seed}/{batch}: exact prediction repetition', np.array_equal(a[1], b[1]))
            check(f'{seed}/{batch}: exact final state repetition', a[2]['final_state_sha256'] == b[2]['final_state_sha256'])
        check(f'{seed}: shared initial state', probabilities[(seed, 16, 1)][2]['start_state_sha256'] == probabilities[(seed, 32, 1)][2]['start_state_sha256'])
    pairs, speeds = [], []
    for seed in [42, 43, 44]:
        per_session = []
        for session in [1, 2]:
            a = next(r for r in rows if (int(r['seed']), int(r['batch']), int(r['session'])) == (seed, 16, session))
            b = next(r for r in rows if (int(r['seed']), int(r['batch']), int(r['session'])) == (seed, 32, session))
            per_session.append(float(a['adaptation_seconds']) / float(b['adaptation_seconds']))
            if session == 1:
                pairs.append(float(b['validation_auroc']) - float(a['validation_auroc']))
        speeds.append(float(np.median(per_session)))
    delta, speed = float(np.mean(pairs)), float(np.median(speeds))
    check('reported AUROC delta', abs(delta - results['mean_validation_auroc_delta']) < 1e-12)
    check('reported median paired speedup', abs(speed - results['median_paired_speedup']) < 1e-12)
    independent = {
        'all_twelve_runs_completed_with_finite_losses': len(rows) == 12,
        'no_skipped_amp_updates': all(int(r['skipped_amp_updates']) == 0 for r in rows),
        'peak_allocated_memory_below_70_percent': max(float(r['peak_allocated_mib']) for r in rows) < .7 * environment['physical_vram_mib'],
        'mean_validation_auroc_drop_at_most_0_005': delta >= -.005,
        'each_seed_validation_auroc_drop_at_most_0_010': min(pairs) >= -.010,
        'median_paired_speedup_at_least_1_10': speed >= 1.10,
    }
    for name, value in independent.items():
        check('acceptance arithmetic: ' + name, bool(value) == results['checks'][name], value)
    check('reported acceptance matches all criteria', results['accepted'] == all(results['checks'].values()))
    epoch_rows = list(csv.DictReader((root / 'epoch_history.csv').open()))
    check('36 finite-loss epochs', len(epoch_rows) == 36 and all(np.isfinite(float(r['train_loss'])) for r in epoch_rows))
    for name, digest in inputs['original_checkpoint_sha256'].items():
        check('original unchanged: ' + name, sha(runtime.PROJECT_ROOT / name) == digest)
    for name, digest in inputs['source_sha256'].items():
        check('versioned source: ' + name, sha(root / 'source' / name) == digest)
    with (root / 'independent_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['check', 'passed', 'detail'])
        writer.writeheader()
        writer.writerows(checks)
    (root / 'INDEPENDENT_REVIEW.md').write_text(
        f"# Independent optimization review\n\nPASS: {len(checks)} checks.\n\nAUROC was recomputed by positive-negative pair counting; F1 by confusion counts. Verified exact repeated validation predictions and state hashes, paired arithmetic, acceptance decisions, finite epoch losses, source snapshots and preservation of original checkpoints. Every file in the original integrity manifest was checked before this review.\n\nA pilot criterion may legitimately fail while this review passes: verification checks the accuracy of the reported outcome. See independent_review.csv for every check.\n", encoding='utf-8')
    # Retain the original training manifest and extend coverage to review outputs.
    training_manifest = root / 'training_integrity.json'
    if not training_manifest.exists():
        training_manifest.write_bytes((root / 'integrity.json').read_bytes())
    (root / 'source' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'integrity.json')
    (root / 'integrity.json').write_text(json.dumps({str(p.relative_to(root)): sha(p) for p in files}, indent=2), encoding='utf-8')
    print(f'PASS: {len(checks)} independent checks; pilot acceptance = {results["accepted"]}.')


if __name__ == '__main__':
    main()
