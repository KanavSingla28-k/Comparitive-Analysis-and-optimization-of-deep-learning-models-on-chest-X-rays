"""Run from the project root: python 04_implementation/src/pipeline.py --help.

No JSON is used to select models or tune hyperparameters: see settings.py.
JSON here records measured evidence, provenance and human-readable explanations.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_ids():
    from settings import CONFIG
    return [f'{name}_{variant}_seed{seed}' for seed in CONFIG['seeds']
            for name in CONFIG['architectures']
            for variant in (['frozen', 'partial'] if name == 'resnet18' else ['partial'])]


def verify(art):
    """Verify data membership, raw predictions, metrics and all checkpoint bytes."""
    from runtime import PROJECT_ROOT, DATA_FILE
    from settings import CONFIG, DATASET_SHA256
    import numpy as np
    from scipy.stats import rankdata
    from experiment import metrics
    freeze = read(PROJECT_ROOT / '05_experiments/configs/protocol_freeze_v1.json')
    for item in freeze['files']:
        require(sha(PROJECT_ROOT / item['relative_path']) == item['sha256'],
                f'Frozen evidence changed: {item["relative_path"]}')
    require(sha(DATA_FILE) == DATASET_SHA256, 'Canonical dataset SHA-256 mismatch')
    with np.load(DATA_FILE, allow_pickle=False) as data:
        accepted = {}
        hashes = {}
        for split in ['train', 'val', 'test']:
            accepted[split] = np.load(PROJECT_ROOT / '03_data/manifests' / f'{split}_indices.npy')
            actual = np.load(art / f'{split}_indices.npy')
            require(np.array_equal(actual, accepted[split]), f'{split} indices differ from accepted membership')
            images = data[f'{split}_images']
            require(images.shape[1:] == (224, 224) and images.dtype == np.uint8, 'Invalid image shape/type')
            hashes[split] = np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in images])
        test_set = set(hashes['test'])
        val = np.flatnonzero([h not in test_set for h in hashes['val']])
        val_set = set(hashes['val'][val])
        train = np.flatnonzero([h not in test_set and h not in val_set for h in hashes['train']])
        for split, expected in [('train', train), ('val', val), ('test', np.arange(624))]:
            require(np.array_equal(expected, accepted[split]), f'Independent cleanup differs: {split}')
        labels = data['test_labels'].reshape(-1)[accepted['test']]
    rows = read(art / 'test_metrics.json')
    expected_ids = run_ids()
    require(len(rows) == 12 and {r['id'] for r in rows} == set(expected_ids), 'Wrong or repeated run matrix')
    results = []
    for rid in expected_ids:
        row = next(r for r in rows if r['id'] == rid)
        history = read(art / 'history' / f'{rid}.json')
        manifest = read(art / 'history' / f'{rid}_manifest.json')
        name, variant, seed_part = rid.rsplit('_', 2)
        require(manifest['id'] == rid and manifest['architecture'] == name and
                manifest['seed'] == int(seed_part[4:]) and manifest['epochs'] == 5, f'Wrong manifest: {rid}')
        require([h['epoch'] for h in history] == [1, 2, 3, 4, 5], f'Incomplete history: {rid}')
        phases = ['head'] * 2 + [variant if variant == 'partial' else 'head'] * 3
        require([h['phase'] for h in history] == phases, f'Wrong training phases: {rid}')
        require(all(np.isfinite(h[k]) for h in history for k in ['train_loss', 'val_auroc', 'val_ap']), f'Non-finite history: {rid}')
        checksum = sha(art / 'checkpoints' / f'{rid}.pt')
        require(checksum == manifest['checkpoint_sha256'], f'Checkpoint bytes differ: {rid}')
        with np.load(art / 'predictions' / f'{rid}.npz', allow_pickle=False) as p:
            require(np.array_equal(p['indices'], accepted['test']), f'Wrong test order: {rid}')
            require(np.array_equal(p['labels'], labels), f'Wrong test labels: {rid}')
            for key in ['probabilities', 'fp32_probabilities']:
                require(p[key].shape == (624,) and np.isfinite(p[key]).all() and
                        ((p[key] >= 0) & (p[key] <= 1)).all(), f'Invalid probabilities: {rid}/{key}')
            measured = metrics(labels, p['probabilities'])
            for metric, value in measured.items():
                require(abs(value - row[metric]) < 1e-12, f'Metric mismatch: {rid}/{metric}')
            positive = labels == 1
            npos = int(positive.sum())
            rank_auc = (rankdata(p['probabilities'])[positive].sum() - npos*(npos+1)/2) / (npos * int((~positive).sum()))
            require(abs(rank_auc - row['auroc']) < 1e-12, f'Independent rank AUROC mismatch: {rid}')
            from sklearn.metrics import roc_auc_score
            require(abs(roc_auc_score(labels, p['fp32_probabilities']) - row['fp32_auroc']) < 1e-12, f'FP32 AUROC mismatch: {rid}')
        results.append({'run_id': rid, 'checkpoint_sha256': checksum, 'status': 'PASS'})
    for seed in CONFIG['seeds']:
        a, b = [read(art / 'history' / f'resnet18_{v}_seed{seed}_manifest.json') for v in ['frozen', 'partial']]
        require(a['shared_epoch2_hash'] == b['shared_epoch2_hash'], 'Unpaired ResNet branches')
        ah, bh = [read(art / 'history' / f'resnet18_{v}_seed{seed}.json') for v in ['frozen', 'partial']]
        require(ah[:2] == bh[:2], 'Shared warm-up histories differ')
    return {'status': 'PASS', 'runs_verified': len(results), 'frozen_files_verified': len(freeze['files']),
            'dataset_sha256': DATASET_SHA256, 'accepted_split_counts': {k: len(v) for k,v in accepted.items()},
            'independent_rank_auroc': True, 'all_predictive_metrics_recomputed': True,
            'test_labels_and_order_verified': True, 'runs': results,
            'description': 'All twelve checkpoint hashes, five-epoch histories, accepted splits and raw prediction-derived scores passed. This checks artifact consistency; checkpoint inference is checked separately by replay.'}


def replay(art):
    """Actually run every saved checkpoint; compare both precisions to saved NPZ."""
    import experiment as e
    import numpy as np
    import torch
    e.setup()
    arrays = e.ImageArrays(['test'])
    rows = []
    for rid in run_ids():
        name = rid.rsplit('_', 2)[0]
        model, _ = e.model_factory(name, pretrained=False)
        model.load_state_dict(torch.load(art / 'checkpoints' / f'{rid}.pt', map_location=e.DEVICE, weights_only=True), strict=True)
        values = {}
        with np.load(art / 'predictions' / f'{rid}.npz') as saved:
            for key, amp in [('probabilities', True), ('fp32_probabilities', False)]:
                actual = e.predict(model, arrays, 'test', amp=amp)
                delta = float(np.max(np.abs(actual - saved[key])))
                require(delta <= 1e-6, f'Checkpoint replay mismatch: {rid}/{key}, {delta}')
                values[key + '_max_absolute_replay_difference'] = delta
        rows.append({'run_id': rid, 'status': 'PASS', **values})
        print('REPLAY PASS', rid, values, flush=True)
        del model
        torch.cuda.empty_cache()
    result = {'status': 'PASS', 'absolute_tolerance': 1e-6, 'runs': rows,
              'description': 'All twelve models were strictly loaded and executed on all 624 real test images in AMP and FP32. These numbers compare replay against the same precision, not AMP against FP32.'}
    save(art / 'checkpoint_replay.json', result)
    return result


def provenance(mode):
    from runtime import PROJECT_ROOT, RUN_ROOT, ARCHIVE
    from settings import PROTOCOL_ID
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_id': PROTOCOL_ID,
              'mode': mode, 'new_training': mode == 'fresh_training',
              'source_evidence': str(ARCHIVE) if mode != 'fresh_training' else 'This versioned run',
              'test_previously_examined': True,
              'source_files': {str(p.relative_to(PROJECT_ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    save(RUN_ROOT / 'provenance.json', result)
    return result


def copy_weights():
    from runtime import PROJECT_ROOT, ARCHIVE
    source = ARCHIVE / '.cache/torch/hub/checkpoints'
    target = PROJECT_ROOT / '.cache/torch/hub/checkpoints'
    target.mkdir(parents=True, exist_ok=True)
    for weight in source.glob('*.pth'):
        destination = target / weight.name
        if not destination.exists():
            shutil.copy2(weight, destination)
        require(sha(weight) == sha(destination), f'Pretrained weight cache mismatch: {weight.name}')


def adopt():
    from runtime import ARCHIVE, RUN_ROOT, ART
    require(not ART.exists(), 'Run artifacts already exist. Choose a new --run-dir; outputs are never silently overwritten.')
    verification = verify(ARCHIVE / 'artifacts')
    ART.mkdir(parents=True)
    for sub in ['checkpoints', 'history', 'predictions', 'benchmarks']:
        shutil.copytree(ARCHIVE / 'artifacts' / sub, ART / sub)
    for name in ['test_metrics.json', 'environment.json', 'dataset_audit.json',
                 'train_indices.npy', 'val_indices.npy', 'test_indices.npy', 'metrics.csv']:
        shutil.copy2(ARCHIVE / 'artifacts' / name, ART / name)
    provenance('saved_evidence_analysis_reproduction')
    save(ART / 'verification.json', verification)
    print('ADOPT PASS: copied and verified twelve existing runs; no new training.', flush=True)


def prepare_fresh():
    from runtime import RUN_ROOT, ART, DATA_FILE, PROJECT_ROOT
    from settings import CONFIG, DATASET_SHA256
    import experiment as e
    import numpy as np
    require(not ART.exists(), 'Use a new --run-dir for a full retraining.')
    require(DATA_FILE.exists() and sha(DATA_FILE) == DATASET_SHA256, 'Dataset missing/corrupt; use download command')
    copy_weights()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    provenance('fresh_training')
    save(RUN_ROOT / 'settings_snapshot.json', CONFIG)
    e.prepare()
    for split in ['train', 'val', 'test']:
        require(np.array_equal(np.load(ART / f'{split}_indices.npy'),
                               np.load(PROJECT_ROOT / '03_data/manifests' / f'{split}_indices.npy')),
                'Fresh audit does not match accepted data policy')


def analyze_and_export():
    from runtime import ART, RUN_ROOT, PROJECT_ROOT
    # Preserve launch-time hashes and separately record sources used to regenerate
    # final analysis. A report correction does not change the trained checkpoints.
    origin = read(RUN_ROOT / 'provenance.json')
    origin['analysis_execution'] = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'description': 'Current analysis/export source hashes. Original launch-time hashes remain in source_files. Session-specific timing interpretation is generated without changing training or saved predictions.',
        'source_files': {str(p.relative_to(PROJECT_ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    save(RUN_ROOT / 'provenance.json', origin)
    import analyze
    from explained_results import export
    verification = verify(ART)
    save(ART / 'verification.json', verification)
    analyze.make_report()
    export()


def download():
    from runtime import DATA_FILE, PROJECT_ROOT
    from settings import CONFIG, DATASET_SHA256
    import urllib.request
    destination = PROJECT_ROOT / '03_data/raw/pneumoniamnist_224.npz'
    if DATA_FILE.exists():
        require(sha(DATA_FILE) == DATASET_SHA256, 'Existing data checksum mismatch')
        print('Canonical dataset is present and verified:', DATA_FILE)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.npz.partial')
    urllib.request.urlretrieve(CONFIG['dataset_url'], partial)
    require(sha(partial) == DATASET_SHA256, 'Downloaded data checksum mismatch')
    partial.replace(destination)
    print('Verified dataset:', destination)
    print('Set IRS_DATA_FILE to this absolute path before the next command.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'reproduce-results', 'full', 'prepare', 'train',
                        'evaluate', 'benchmark', 'hardware', 'replay', 'analyze', 'download'])
    parser.add_argument('--run-dir', default='05_experiments/reproduced_v1', help='Child path inside this project / 05_experiments')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    candidate = Path(args.run_dir)
    os.environ['IRS_RUN_DIR'] = str((project / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    from runtime import RUN_ROOT, ART, ARCHIVE
    import experiment as e
    if args.command == 'verify':
        result = verify(ARCHIVE / 'artifacts')
        output = project / '06_results/reproducibility_checks.json'
        save(output, result)
        output.with_suffix('.md').write_text('# Artifact verification\n\n' + result['description'] + '\n\nTwelve runs and eighteen frozen files passed. See the executable verification in `pipeline.py`.\n', encoding='utf-8')
        print('VERIFY PASS', output)
    elif args.command == 'download':
        download()
    elif args.command == 'reproduce-results':
        adopt()
        replay(ART)
        analyze_and_export()
    elif args.command == 'full':
        prepare_fresh()
        e.train(e.CONFIG['seeds'])
        e.evaluate()
        replay(ART)
        e.benchmark()
        from hardware_probe import probe
        probe()
        analyze_and_export()
    elif args.command == 'prepare':
        prepare_fresh()
    elif args.command == 'train':
        require(read(RUN_ROOT / 'provenance.json')['new_training'], 'Training is only allowed in a prepared fresh run')
        lock = read(ART / 'protocol_lock.json')
        require(lock['config'] == e.CONFIG and lock['source_sha256'] == sha(Path(e.__file__)), 'Prepared source/settings changed')
        e.train(e.CONFIG['seeds'])
    elif args.command == 'evaluate':
        e.evaluate()
    elif args.command == 'benchmark':
        e.benchmark()
    elif args.command == 'hardware':
        from hardware_probe import probe
        probe()
    elif args.command == 'replay':
        replay(ART)
    elif args.command == 'analyze':
        analyze_and_export()
    print('COMPLETE', args.command, RUN_ROOT, flush=True)


if __name__ == '__main__':
    main()