"""Independently compare the delivered reproduction against the preserved pilot.

Run after both complete: python -B 04_implementation/src/validate_reproduction.py
This records equality in THIS tested environment, not a cross-platform promise.
"""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'archive/pilot_2026-10-22/.deps'))
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate():
    archive = PROJECT / 'archive/pilot_2026-10-22/artifacts'
    regenerated = PROJECT / '05_experiments/reproduced_v1/artifacts'
    fresh = PROJECT / '05_experiments/retrained_v1/artifacts'
    original_rows = load(archive / 'test_metrics.json')
    rows = []
    for original in original_rows:
        rid = original['id']
        checkpoint_equal = sha(archive / 'checkpoints' / f'{rid}.pt') == sha(fresh / 'checkpoints' / f'{rid}.pt')
        check(checkpoint_equal, f'Fresh checkpoint differs from pilot: {rid}')
        original_npz = np.load(archive / 'predictions' / f'{rid}.npz')
        fresh_npz = np.load(fresh / 'predictions' / f'{rid}.npz')
        checks = {k: bool(np.array_equal(original_npz[k], fresh_npz[k])) for k in original_npz.files}
        check(all(checks.values()), f'Fresh probabilities/membership differ: {rid}')
        rows.append({'run_id': rid, 'checkpoint_bytes_identical_to_pilot': checkpoint_equal,
                     'prediction_arrays_identical_to_pilot': checks})
    old_stats = load(archive / 'statistical_summary.json')
    regenerated_stats = load(regenerated / 'statistical_summary.json')
    fresh_stats = load(fresh / 'statistical_summary.json')
    # Resource times are expected to change. Only predictive/statistical quantities
    # and exact bootstrap arrays are equality targets here.
    for item in old_stats['summary']:
        for candidate in [regenerated_stats, fresh_stats]:
            actual = next(r for r in candidate['summary'] if r['group'] == item['group'])
            for key in ['auroc_ci95'] + [k for k in item if k.endswith('_mean') or k.endswith('_sd')]:
                if key == 'training_minutes_mean':
                    continue
                check(actual[key] == item[key], f'Predictive summary mismatch: {item["group"]}/{key}')
    for candidate in [regenerated_stats, fresh_stats]:
        check(candidate['paired_comparisons'] == old_stats['paired_comparisons'], 'Paired statistics changed')
    for art in [regenerated, fresh]:
        with np.load(archive / 'bootstrap_samples.npz') as a, np.load(art / 'bootstrap_samples.npz') as b:
            check(np.array_equal(a['mean_auroc'], b['mean_auroc']), 'Bootstrap arrays changed')
        replay = load(art / 'checkpoint_replay.json')
        check(replay['status'] == 'PASS' and len(replay['runs']) == 12, 'Checkpoint replay not complete')
    for name in ['reproduced_v1', 'retrained_v1']:
        bundle = PROJECT / '06_results' / name
        explained = load(bundle / 'results_explained.json')
        check(len(explained['comparison']) == 4 and len(explained['per_run_results']) == 12, 'Incomplete explained JSON')
        check(all(r.get('description') for r in explained['comparison'] + explained['paired_differences'] + explained['optimization_and_hardware'] + explained['per_run_results']), 'Missing result descriptions')
        for line in (bundle / 'SHA256SUMS.txt').read_text().splitlines():
            digest, relative = line.split('  ', 1)
            check(sha(bundle / relative) == digest, f'Export hash mismatch: {relative}')
        report = (bundle / 'comparative_analysis_report.md').read_text(encoding='utf-8')
        import re
        for path in re.findall(r'!\[[^]]*\]\(([^)]+)\)', report):
            check((bundle / path).is_file(), f'Broken figure link: {path}')
    freeze = load(PROJECT / '05_experiments/configs/protocol_freeze_v1.json')
    check(all(sha(PROJECT / f['relative_path']) == f['sha256'] for f in freeze['files']), 'Frozen archive/protocol changed')
    source_checks = []
    for run in ['reproduced_v1', 'retrained_v1']:
        manifest = load(PROJECT / '05_experiments' / run / 'provenance.json')
        current_sources = dict(manifest['source_files'])
        current_sources.update(manifest.get('analysis_execution', {}).get('source_files', {}))
        check(all(sha(PROJECT / path) == digest for path, digest in current_sources.items()), 'Recorded active source was modified after its latest execution')
        source_checks.append(run)
    result = {'status': 'PASS', 'fresh_training_runs': len(rows),
              'checkpoint_bytes_and_prediction_arrays': rows,
              'predictive_summaries_and_paired_intervals_identical': True,
              'all_2000_bootstrap_draws_identical': True,
              'export_hashes_figure_links_and_descriptions_checked': True,
              'frozen_files_unchanged': len(freeze['files']),
              'active_sources_match_execution_provenance': source_checks,
              'validator_sha256': sha(Path(__file__)),
              'description': 'In the tested Windows/CUDA environment, full retraining reproduced all twelve original checkpoint files and all saved test-prediction arrays exactly. Saved-evidence and fresh analyses reproduced every bootstrap draw, predictive summary and paired interval. Hardware timings were remeasured and are not equality targets.'}
    output = PROJECT / '06_results/reproduction_validation.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(result['description'])
    print('PASS:', output)


if __name__ == '__main__':
    validate()