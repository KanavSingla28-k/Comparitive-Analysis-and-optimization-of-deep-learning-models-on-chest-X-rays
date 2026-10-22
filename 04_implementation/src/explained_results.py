"""Export actual measurements with explanations in the SAME JSON document.

CSV and Markdown are parallel supervisor-facing outputs, not replacements for code.
"""
import csv
from pathlib import Path
import shutil
from runtime import PROJECT_ROOT, RUN_ROOT, ART
from settings import CONFIG, PROTOCOL_ID
from pipeline import read, save, sha, run_ids

DEFINITIONS = {
    'auroc': 'Area under ROC: ranking of pneumonia above normal across thresholds; higher is better; chance is 0.5.',
    'ap': 'Average precision: stepwise precision-recall summary, not trapezoidal PR area; test prevalence baseline is 0.625.',
    'f1': 'Harmonic mean of pneumonia precision and recall at probability >= 0.5.',
    'precision': 'TP/(TP+FP): proportion of predicted pneumonia examples labeled pneumonia.',
    'recall': 'TP/(TP+FN): pneumonia sensitivity.',
    'specificity': 'TN/(TN+FP): normal-example sensitivity.',
    'accuracy': '(TP+TN)/N: fraction correct; majority-label reference is 0.625.',
    'balanced_accuracy': 'Mean of sensitivity and specificity.',
    'sd': 'Sample standard deviation across three training seeds (ddof=1); not standard error or patient uncertainty.',
    'auroc_ci95': 'Exploratory percentile interval: 2000 paired seed-and-exact-image-cluster bootstrap draws.',
    'speedup': 'FP32 time divided by AMP time; below 1 means AMP was slower.',
    'peak_allocated_mib': 'Peak live tensor allocation in MiB; excludes unused reserved memory and driver/desktop allocations.',
}


def write_csv(path, rows):
    keys = list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export():
    stats = read(ART / 'statistical_summary.json')
    raw = read(ART / 'test_metrics.json')
    origin = read(RUN_ROOT / 'provenance.json')
    verification = read(ART / 'verification.json')
    models = []
    for item in stats['summary']:
        record = dict(item)
        record['description'] = (
            f"{item['label']} had mean test AUROC {item['auroc_mean']:.6f} "
            f"with seed SD {item['auroc_sd']:.6f}; F1 at 0.5 was {item['f1_mean']:.6f}. "
            'These are means of three separate model scores, not an ensemble. '
            'This descriptive result does not establish superiority over another architecture.')
        record['evidence_runs'] = [f"{item['group']}_seed{s}" for s in CONFIG['seeds']]
        models.append(record)
    pairs = []
    for pair in stats['paired_comparisons']:
        lo, hi = pair['ci95']
        verdict = ('supports a positive within-benchmark difference' if lo > 0 else
                   'supports a negative within-benchmark difference' if hi < 0 else
                   'includes zero, so a reliable directional difference is not established')
        pairs.append({**pair, 'description':
                      f"Mean AUROC difference {pair['delta_auroc_mean']:+.6f}; "
                      f"exploratory 95% interval [{lo:+.6f}, {hi:+.6f}] {verdict}. "
                      'Three seeds and unknown patient dependence limit this inference; no multiplicity adjustment was applied.'})
    hardware = []
    for name in CONFIG['architectures']:
        bench = stats['benchmark'][name]
        rows = [r for r in stats['synthetic_training_capacity'] if r['architecture'] == name and 'error' not in r]
        fp = next(r for r in rows if r['batch_size'] == 16 and not r['amp'])
        amp = next(r for r in rows if r['batch_size'] == 16 and r['amp'])
        acceleration = fp['median_step_ms'] / amp['median_step_ms']
        memory_saved = 100 * (1 - amp['peak_allocated_mib'] / fp['peak_allocated_mib'])
        hardware.append({'architecture': name, 'batch_1_inference': bench,
                         'synthetic_full_gradient_training': rows,
                         'batch_16_training_amp_speedup': acceleration,
                         'batch_16_training_allocation_reduction_percent': memory_saved,
                         'description':
                         f"Batch-1 AMP forward speedup was {bench['speedup_median']:.3f}x (below 1 is slower). "
                         f"At synthetic full-gradient batch 16, AMP step speedup was {acceleration:.3f}x "
                         f"and allocated memory fell by {memory_saved:.1f}%. "
                         f"Largest successful tested full-gradient batch: {max(r['batch_size'] for r in rows)}. "
                         'Timing inputs had matching shapes but fresh random values per inference case. '
                         'Full-gradient probes used disposable model copies and synthetic targets, not additional predictive training. '
                         'These short device-specific probes measure feasible tested workloads, not a guaranteed training gain or absolute VRAM ceiling.'})
    per_run = []
    flat_history = []
    for row in raw:
        rid = row['id']
        history = read(ART / 'history' / f'{rid}.json')
        for epoch in history:
            flat_history.append({'run_id': rid, **epoch})
        per_run.append({**row, 'history': history,
                        'checkpoint': str((ART / 'checkpoints' / f'{rid}.pt').relative_to(PROJECT_ROOT)),
                        'predictions': str((ART / 'predictions' / f'{rid}.npz').relative_to(PROJECT_ROOT)),
                        'description': f"Seed {row['seed']} {row['architecture']} {row['variant']}: "
                        f"AUROC {row['auroc']:.6f}, F1 {row['f1']:.6f} on all 624 test images. "
                        f"At threshold 0.5: TN={row['tn']}, FP={row['fp']}, FN={row['fn']}, TP={row['tp']}. "
                        f"AMP versus FP32 changed {row['amp_fp32_decision_disagreements']} threshold decisions. "
                        'Epoch loss is weighted training BCE; validation BCE is unweighted.'})
    result = {
        'schema_version': 1, 'protocol_id': PROTOCOL_ID,
        'description': 'Reproducible educational comparison of three pretrained CNNs for pediatric pneumonia classification. All descriptions below are generated from the stored numerical evidence. Raw predictions, checkpoints, executable Python and CSV tables accompany this JSON.',
        'provenance': origin,
        'pipeline': [
            {'stage': 'audit', 'description': 'Check official dataset checksum and exact image duplicates; retain 4698/524/624 examples.'},
            {'stage': 'train', 'description': 'For each of three seeds, train each head for 2 epochs and partially fine-tune for 3; paired ResNet frozen control shares epoch-2 weights. Batch 16, AMP, fixed BN statistics.'},
            {'stage': 'evaluate', 'description': 'Execute final checkpoints on the same test order in AMP and FP32; fixed threshold 0.5; save raw arrays.'},
            {'stage': 'statistics', 'description': 'Recompute per-run scores, sample SD and paired hierarchical bootstrap from saved predictions; never tune on test scores.'},
            {'stage': 'hardware', 'description': 'Time isolated forward calls and synthetic full-gradient steps with explicit warm-ups, synchronization and raw samples.'},
            {'stage': 'export', 'description': 'Write explanations with numerical results in this JSON, plus CSV, Markdown and PNG figures.'}],
        'metric_definitions': DEFINITIONS, 'settings': CONFIG,
        'verification': verification,
        'checkpoint_replay': read(ART / 'checkpoint_replay.json') if (ART / 'checkpoint_replay.json').exists() else {'status': 'not_run', 'description': 'Run the replay command to check checkpoint-to-prediction execution; byte integrity alone is not replay.'},
        'dataset': {'accepted_counts': {'train': 4698, 'val': 524, 'test': 624},
                    'test_class_counts': {'normal': 234, 'pneumonia': 390},
                    'description': 'PneumoniaMNIST+ actual 224 resolution; pediatric normal/pneumonia. Ten exact train/validation-overlap rows removed. Within-test duplicate images retained and grouped for bootstrap. No patient IDs; exact hash checks cannot prove patient independence.'},
        'comparison': models, 'paired_differences': pairs,
        'statistics_method': stats['bootstrap'], 'optimization_and_hardware': hardware,
        'per_run_results': per_run,
        'limitations': ['Previously examined test set; no prospective blinded claim.',
                        'Single pediatric binary dataset; no adult multi-finding or clinical deployment conclusion.',
                        'Only three seeds; exact-image clusters do not capture unknown patient clusters.',
                        'Architecture comparisons are exploratory; inconclusive differences do not prove equivalence.',
                        'Hardware timings depend on device/software/session; fixed five-epoch training does not establish convergence.',
                        'Fresh runs are method reproductions; bitwise equality across hardware/software is not guaranteed.'],
    }
    save(RUN_ROOT / 'results_explained.json', result)
    output = PROJECT_ROOT / '06_results' / RUN_ROOT.name
    output.mkdir(parents=True, exist_ok=True)
    for name in ['results_explained.json', 'comparative_analysis_report.md', 'provenance.json']:
        shutil.copy2(RUN_ROOT / name, output / name)
    for name in ['comparison_summary.csv', 'metrics.csv']:
        shutil.copy2(ART / name, output / name)
    write_csv(output / 'epoch_history.csv', flat_history)
    write_csv(output / 'paired_differences.csv', [{
        'comparison': p['comparison'], 'delta_auroc_mean': p['delta_auroc_mean'],
        'ci95_low': p['ci95'][0], 'ci95_high': p['ci95'][1], 'description': p['description']} for p in pairs])
    hardware_rows = []
    for case in hardware:
        b = case['batch_1_inference']
        hardware_rows.append({'architecture': case['architecture'],
                              'fp32_batch1_median_ms': b['fp32']['median_ms'],
                              'amp_batch1_median_ms': b['amp']['median_ms'],
                              'batch1_amp_speedup': b['speedup_median'],
                              'batch16_full_gradient_amp_speedup': case['batch_16_training_amp_speedup'],
                              'batch16_allocation_reduction_percent': case['batch_16_training_allocation_reduction_percent']})
    write_csv(output / 'hardware_summary.csv', hardware_rows)
    shutil.copytree(ART / 'figures', output / 'figures', dirs_exist_ok=True)
    report = (output / 'comparative_analysis_report.md').read_text(encoding='utf-8')
    report = report.replace((ART / 'figures').as_posix(), 'figures')
    (output / 'comparative_analysis_report.md').write_text(report, encoding='utf-8')
    paths = [p for p in output.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt']
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(output).as_posix()}\n' for p in sorted(paths)), encoding='utf-8')
    print('EXPLAINED RESULTS AND READABLE REPORT:', output, flush=True)