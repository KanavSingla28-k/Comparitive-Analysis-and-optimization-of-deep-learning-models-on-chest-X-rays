"""Publish verified batch-size pilot into the report and refresh package integrity.

Run only after optimize.ps1 succeeds. Does not train or change v1 model evidence.
"""
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from batch32_pilot import sha, verify

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / '05_experiments' / 'batch32_v2_pilot'


def text(path, content):
    Path(path).write_text(content, encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_section(path, marker, content):
    original = Path(path).read_text(encoding='utf-8-sig')
    original = original.split(marker)[0].rstrip()
    text(path, original + '\n\n' + marker + '\n\n' + content.rstrip() + '\n')


def main():
    verify(RUN)
    checks = list(csv.DictReader((RUN / 'independent_review.csv').open()))
    assert checks and all(r['passed'] == 'True' for r in checks)
    result = json.loads((RUN / 'results_explained.json').read_text())
    stamp = datetime.now(timezone.utc).isoformat()
    rows = result['summary']
    table = ['| Physical batch | Validation AUROC mean +/- seed SD | Mean validation F1 | Median adaptation seconds | Max allocated MiB |', '| --- | --- | --- | --- | --- |']
    for r in rows:
        table.append(f"| {r['batch']} | {r['mean_validation_auroc']:.6f} +/- {r['seed_sd_validation_auroc']:.6f} | {r['mean_validation_f1']:.6f} | {r['median_adaptation_seconds_technical_runs']:.3f} | {r['max_allocated_mib']:.1f} |")
    table = '\n'.join(table)
    interpretation = (
        f"{result['recommendation']} Median paired speedup was **{result['median_paired_speedup']:.3f}x**, "
        f"equivalent to **{(1 - 1 / result['median_paired_speedup']) * 100:.2f}% less measured adaptation time**. "
        f"Mean paired validation AUROC changed by **{result['mean_validation_auroc_delta']:+.6f}** "
        f"({result['mean_validation_auroc_delta'] * 100:+.4f} percentage points). "
        f"All {len(checks)} independent verification checks passed."
    )
    limits = ('These are validation results for ResNet-18 only, with three seeds and two technical timing sessions per seed/settings pair. '
              'Timing includes three adaptation epochs and per-epoch validation, excluding model loading, warm-up and checkpoint saving. '
              'Two sessions reproduce the same trained states and do not increase the number of independent seeds. '
              'Batch 32 halves optimizer updates from 882 to 441 over the same 14,094 training-image presentations. '
              'V2 pre-generates paired rotations for the full epoch and therefore reruns its own batch-16 control. '
              'Do not compare these validation scores directly to the earlier v1 test table. '
              'No new test evaluation or statistical equivalence claim is made. The v1 test set remains previously inspected. '
              'The gain does not change model parameter count or establish a hardware optimization ceiling.')
    summary = '# Batch-size optimization — completed 19 November 2026\n\n' + interpretation + '\n\n' + table + '\n\n' + limits + '\n\n'
    summary += ('Primary evidence: [full method and results](../05_experiments/batch32_v2_pilot/OPTIMIZATION_REPORT.md), '
                '[per-run CSV](../05_experiments/batch32_v2_pilot/runs.csv), '
                '[paired comparisons](../05_experiments/batch32_v2_pilot/paired_comparisons.csv), '
                '[independent review](../05_experiments/batch32_v2_pilot/INDEPENDENT_REVIEW.md), '
                '[explained JSON](../05_experiments/batch32_v2_pilot/results_explained.json), '
                '[runner](../04_implementation/src/batch32_pilot.py).\n\n'
                '![Measured batch-size comparison](../05_experiments/batch32_v2_pilot/figures/batch_size_comparison.png)\n\n'
                'Reproduce with `./optimize.ps1 -RunDir 05_experiments/batch32_repeat_new`. '
                'Verify the retained pilot with `./optimize.ps1 -Stage verify`. '
                'Review its statistics with `./optimize.ps1 -Stage review`. '
                'The default retained run directory is never overwritten.\n')
    text(PROJECT / '06_results' / 'BATCH_SIZE_OPTIMIZATION.md', summary)
    report_content = ('Executed 19 November 2026 under protocol `cxr_batch32_v2_pilot`.\n\n' + interpretation + '\n\n' + table + '\n\n' + limits + '\n\n'
                      'See [the optimization report](../06_results/BATCH_SIZE_OPTIMIZATION.md), '
                      '[all pilot results](../05_experiments/batch32_v2_pilot/OPTIMIZATION_REPORT.md), '
                      '[independent verification](../05_experiments/batch32_v2_pilot/INDEPENDENT_REVIEW.md) '
                      'and [reproducible implementation](../04_implementation/src/batch32_pilot.py).')
    replace_section(PROJECT / '07_report' / 'FINAL_COMPARATIVE_REPORT.md', '## 13. Completed batch-size optimization extension', report_content)
    report_json_path = PROJECT / '07_report' / 'final_results_explained.json'
    report_json = json.loads(report_json_path.read_text(encoding='utf-8-sig'))
    report_json['batch_size_optimization_v2'] = {'published_utc': stamp,
                                               'evidence': '05_experiments/batch32_v2_pilot',
                                               'independent_checks_passed': len(checks),
                                               'description': 'Separate validation-only extension; original v1 test results above are retained.',
                                               'results': result}
    text(report_json_path, json.dumps(report_json, indent=2, allow_nan=False))
    status = PROJECT / '06_results' / 'OPTIMIZATION_EXECUTION_STATUS.md'
    old_status = PROJECT / '06_results' / 'OPTIMIZATION_INITIAL_RUNTIME_BLOCK.md'
    if not old_status.exists():
        shutil.copy2(status, old_status)
    text(status, '# Optimization execution status — completed 19 November 2026\n\n'
         'The previous Windows PyTorch import rejection no longer occurred on retry. No security settings were changed by this project. '
         'The full project import check succeeded with bundled NumPy 2.5.4, PyTorch 2.10.0+cu128 and CUDA on the RTX 4050. '
         'All twelve pilot runs and independent review completed.\n\n' + interpretation + '\n\n'
         'Read [the completed optimization report](BATCH_SIZE_OPTIMIZATION.md). '
         'The initial diagnostic is retained in [the historical blocker note](OPTIMIZATION_INITIAL_RUNTIME_BLOCK.md).\n\n'
         'Run `./optimize.ps1 -Stage verify` from the project root to verify saved pilot evidence; '
         'use a fresh `-RunDir` to train again.\n')
    replace_section(PROJECT / 'MILESTONE_TRACKER.md', '## Optimization extension — accepted execution 19 November 2026',
                    'COMPLETE: twelve real GPU runs, saved checkpoints and validation predictions, independent numerical review and explained CSV/Markdown/JSON outputs.\n\n' + interpretation + '\n\nSee `06_results/BATCH_SIZE_OPTIMIZATION.md`. Earlier blocker notes are historical; they do not describe the current runtime state.')
    readme = PROJECT / 'README.md'
    original = readme.read_text(encoding='utf-8-sig')
    original = original.split('## Additional batch-size optimization pilot')[0].rstrip()
    text(readme, original + '\n\n## Additional batch-size optimization pilot\n\n'
         'Completed 19 November 2026. ' + interpretation + '\n\n'
         'Read [the completed optimization report](06_results/BATCH_SIZE_OPTIMIZATION.md). '
         '`./optimize.ps1 -Stage verify` checks retained pilot files; `./optimize.ps1 -RunDir 05_experiments/batch32_repeat_new` '
         'runs a new validation-only comparison. CSV, Markdown, explained JSON, source, checkpoints and predictions are retained.\n')
    replace_section(PROJECT / '08_handoff' / 'START_HERE.md', '## Optimization extension completed — 19 November 2026',
                    interpretation + '\n\nSee [the report](../06_results/BATCH_SIZE_OPTIMIZATION.md). '
                    'Earlier references to this pilot as unexecuted are historical. Use `./optimize.ps1 -Stage verify` '
                    'for its dedicated integrity check. Keep the original three-model test comparison separate from these validation results.')
    # Refresh the whole-package snapshot, retaining the earlier manifest as history.
    handoff = PROJECT / '08_handoff'
    archive = handoff / 'previous_package_2026-10-22'
    archive.mkdir(exist_ok=True)
    for name in ['package_manifest.csv', 'package_verification.json', 'package_verification.md']:
        if (handoff / name).exists() and not (archive / name).exists():
            shutil.copy2(handoff / name, archive / name)
    frozen = json.loads((PROJECT / '05_experiments' / 'configs' / 'protocol_freeze_v1.json').read_text())
    for item in frozen['files']:
        assert sha(PROJECT / item['relative_path']) == item['sha256'], 'Frozen v1 source/evidence changed'
    exclusions = {'08_handoff/package_manifest.csv', '08_handoff/package_verification.json', '08_handoff/package_verification.md', '08_handoff/finalize.log'}
    entries = []
    for path in sorted(PROJECT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT)
        if any(p in {'.git', '.venv', '__pycache__'} for p in relative.parts) or path.suffix in {'.pyc', '.tmp'}:
            continue
        if relative.parts[0] == '.cache' or relative.as_posix() in exclusions:
            continue
        assert path.resolve().is_relative_to(PROJECT)
        entries.append({'relative_path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path),
                        'role': 'preserved_pilot_and_offline_dependencies' if relative.parts[0] == 'archive' else 'active_evidence'})
    csv_write(handoff / 'package_manifest.csv', entries)
    # Independent second read of every retained file after manifest creation.
    failures = [r['relative_path'] for r in entries if sha(PROJECT / r['relative_path']) != r['sha256'] or (PROJECT / r['relative_path']).stat().st_size != r['bytes']]
    verification = {'status': 'FAIL' if failures else 'PASS', 'created_utc': stamp, 'files_checked': len(entries),
                    'manifest_sha256': sha(handoff / 'package_manifest.csv'), 'failures': failures,
                    'description': 'Current retained files checked by size and SHA-256, including the optimization extension. Previous manifest retained separately. Git, local venv, active caches, bytecode and current manifest/verification outputs excluded. Extra files created later are not authenticated.'}
    text(handoff / 'package_verification.json', json.dumps(verification, indent=2))
    text(handoff / 'package_verification.md', '# Refreshed package integrity\n\n' + verification['status'] + f": {len(entries)} files checked.\n\n" + verification['description'] + '\n')
    assert not failures, failures
    print(f'PUBLISHED: pilot accepted={result["accepted"]}; independent checks={len(checks)}; package PASS={len(entries)} files.')


if __name__ == '__main__':
    main()
