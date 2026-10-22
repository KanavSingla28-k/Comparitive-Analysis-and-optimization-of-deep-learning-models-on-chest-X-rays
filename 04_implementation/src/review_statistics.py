"""Independent Phase 5 checks from raw predictions and raw timing samples.

This module does not train models or call the original bootstrap implementation.
The bootstrap AUC uses weighted positive/negative pair counting with tie handling.
"""
import csv
import hashlib
import json
from pathlib import Path
from runtime import PROJECT_ROOT, DATA_FILE
import numpy as np
from sklearn.metrics import average_precision_score
from settings import CONFIG, GROUPS, LABELS, PAIRED_COMPARISONS, BOOTSTRAP_SEED

OUT = PROJECT_ROOT / '06_results'
SESSIONS = ['reproduced_v1', 'retrained_v1']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csv_out(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Checks:
    def __init__(self):
        self.rows = []

    def close(self, name, actual, expected, tol=1e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error = float(np.max(np.abs(a-b))) if ok else float('inf')
        ok = ok and error <= tol
        self.rows.append({'check': name, 'status': 'PASS' if ok else 'FAIL',
                          'maximum_absolute_difference': error, 'tolerance': tol})
        if not ok:
            raise RuntimeError(f'{name}: discrepancy {error}, tolerance {tol}')


def independent_auc(y, p, weights=None):
    """Probability that a positive outranks a negative, counting ties as half."""
    weights = np.ones(len(y), dtype=np.int64) if weights is None else weights
    _, level = np.unique(p, return_inverse=True)
    positive = np.bincount(level, weights=weights*(y == 1))
    negative = np.bincount(level, weights=weights*(y == 0))
    denominator = positive.sum() * negative.sum()
    if denominator == 0:
        raise RuntimeError('Bootstrap sample contains only one class')
    negatives_below = negative.cumsum() - negative
    return float(np.dot(positive, negatives_below + .5*negative) / denominator)


def metrics(y, p):
    decision = p >= .5
    tn = int(((y == 0) & ~decision).sum())
    fp = int(((y == 0) & decision).sum())
    fn = int(((y == 1) & ~decision).sum())
    tp = int(((y == 1) & decision).sum())
    recall, specificity = tp/(tp+fn), tn/(tn+fp)
    return {'auroc': independent_auc(y,p), 'ap': float(average_precision_score(y,p)),
            'f1': 2*tp/(2*tp+fp+fn), 'precision': tp/(tp+fp) if tp+fp else 0.,
            'recall': recall, 'specificity': specificity,
            'accuracy': (tp+tn)/len(y), 'balanced_accuracy': (recall+specificity)/2,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


def independent_bootstrap(probabilities, y):
    with np.load(DATA_FILE) as data:
        hashes = [hashlib.sha256(image.tobytes()).hexdigest() for image in data['test_images']]
    group_ids = {value: index for index, value in enumerate(dict.fromkeys(hashes))}
    membership = np.array([group_ids[h] for h in hashes])
    if len(group_ids) != 621:
        raise RuntimeError('Expected 621 exact-image groups')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty((CONFIG['bootstrap_repetitions'], len(GROUPS)))
    for iteration in range(len(samples)):
        drawn = rng.integers(0, len(group_ids), len(group_ids))
        weights = np.bincount(drawn, minlength=len(group_ids))[membership]
        seed_indices = rng.integers(0, 3, 3)
        # Repeat cases through integer weights rather than materializing the
        # resampled array and applying the original rank-sum implementation.
        for model in range(len(GROUPS)):
            scores = np.array([independent_auc(y, probabilities[model,s], weights) for s in range(3)])
            samples[iteration,model] = scores[seed_indices].mean()
    return samples


def review():
    checks = Checks()
    result = {'status': 'PASS', 'description': 'Independent raw-prediction, weighted-pair bootstrap and raw-timing verification for both retained sessions; no new training or test-driven tuning.', 'sessions': {}}
    accepted_summary, precision_rows, hardware_rows = [], [], []
    independent_draws = None
    for session in SESSIONS:
        art = PROJECT_ROOT / '05_experiments' / session / 'artifacts'
        stats = read(art / 'statistical_summary.json')
        recorded = {r['id']: r for r in read(art / 'test_metrics.json')}
        probabilities = np.empty((4,3,624))
        summaries = {r['group']: r for r in stats['summary']}
        all_metrics = {}
        for i, group in enumerate(GROUPS):
            all_metrics[group] = []
            precision_deltas, disagreements, auc_deltas = [], [], []
            for j, seed in enumerate(CONFIG['seeds']):
                rid = f'{group}_seed{seed}'
                with np.load(art / 'predictions' / f'{rid}.npz') as data:
                    y = data['labels'].astype(int)
                    p, fp32 = data['probabilities'], data['fp32_probabilities']
                    probabilities[i,j] = p
                    computed = metrics(y,p)
                    for key, value in computed.items():
                        checks.close(f'{session}/{rid}/{key}', value, recorded[rid][key])
                    delta = float(np.max(np.abs(p-fp32)))
                    flips = int(((p >= .5) != (fp32 >= .5)).sum())
                    fp32_auc = independent_auc(y, fp32)
                    checks.close(f'{session}/{rid}/precision_probability_difference', delta, recorded[rid]['amp_fp32_max_probability_difference'])
                    checks.close(f'{session}/{rid}/precision_decisions', flips, recorded[rid]['amp_fp32_decision_disagreements'])
                    checks.close(f'{session}/{rid}/fp32_auroc', fp32_auc, recorded[rid]['fp32_auroc'])
                    precision_deltas.append(delta)
                    disagreements.append(flips)
                    auc_deltas.append(abs(computed['auroc']-fp32_auc))
                    all_metrics[group].append(computed)
            summary = summaries[group]
            for metric in ['auroc','ap','f1','precision','recall','specificity','accuracy','balanced_accuracy']:
                values = [m[metric] for m in all_metrics[group]]
                checks.close(f'{session}/{group}/{metric}_mean', np.mean(values), summary[metric+'_mean'])
                checks.close(f'{session}/{group}/{metric}_sd', np.std(values,ddof=1), summary[metric+'_sd'])
            precision_rows.append({'session': session, 'group': group,
                                   'maximum_probability_difference': max(precision_deltas),
                                   'threshold_disagreements_over_1872_seed_predictions': sum(disagreements),
                                   'maximum_absolute_auroc_difference': max(auc_deltas),
                                   'description': 'Same real test images and checkpoint; precision difference is not checkpoint replay error.'})
            manifests = [read(art/'history'/f'{group}_seed{s}_manifest.json') for s in CONFIG['seeds']]
            checks.close(f'{session}/{group}/weights_mib', np.mean([m['checkpoint_bytes_fp32']/2**20 for m in manifests]), summary['weights_mib'])
            checks.close(f'{session}/{group}/training_minutes', np.mean([m['total_seconds_including_validation']/60 for m in manifests]), summary['training_minutes_mean'])
            checks.close(f'{session}/{group}/training_peak', max(d['peak_allocated_mib_including_data'] for m in manifests for d in m['phase_details']), summary['training_peak_mib'])
        if independent_draws is None:
            independent_draws = independent_bootstrap(probabilities,y)
            destination = OUT / 'statistics/independent_bootstrap_samples.npz'
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, mean_auroc=independent_draws)
        saved_draws = np.load(art / 'bootstrap_samples.npz')['mean_auroc']
        checks.close(f'{session}/all_2000_independent_bootstrap_draws', independent_draws, saved_draws, tol=1e-14)
        for i, group in enumerate(GROUPS):
            checks.close(f'{session}/{group}/auroc_ci95', np.percentile(independent_draws[:,i],[2.5,97.5]), summaries[group]['auroc_ci95'])
            if session == 'retrained_v1':
                item = dict(summaries[group])
                item.pop('seeds')
                low, high = item.pop('auroc_ci95')
                accepted_summary.append({'role': 'paired_control' if i == 0 else 'core_comparison', **item,
                                         'auroc_ci95_low': low, 'auroc_ci95_high': high,
                                         'description': 'Mean of three separately trained model scores; SD is across seeds; confidence interval is exploratory.'})
        for k,(a,b) in enumerate(PAIRED_COMPARISONS):
            pair = stats['paired_comparisons'][k]
            checks.close(f'{session}/paired_difference_{k}', summaries[GROUPS[a]]['auroc_mean']-summaries[GROUPS[b]]['auroc_mean'], pair['delta_auroc_mean'])
            checks.close(f'{session}/paired_ci95_{k}', np.percentile(independent_draws[:,a]-independent_draws[:,b],[2.5,97.5]), pair['ci95'])
        for row in read(art / 'benchmarks/inference.json'):
            if 'error' in row:
                continue
            timings = np.asarray(row['timings_ms'])
            prefix = f'{session}/inference/{row["architecture"]}/{row["batch_size"]}/{row["amp"]}/{row["round"]}'
            checks.close(prefix+'/sample_count', len(timings), row['repeats'])
            checks.close(prefix+'/median_ms', np.median(timings), row['median_ms'])
            checks.close(prefix+'/mean_ms', np.mean(timings), row['mean_ms'])
            checks.close(prefix+'/p95_ms', np.percentile(timings,95), row['p95_ms'])
            if np.any(timings <= 0) or row['wall_images_per_second'] <= 0:
                raise RuntimeError('Invalid timing/throughput')
        training = read(art / 'benchmarks/training_capacity.json')
        for row in training:
            if 'error' in row:
                continue
            prefix = f'{session}/synthetic/{row["architecture"]}/{row["batch_size"]}/{row["amp"]}'
            checks.close(prefix+'/sample_count', len(row['times_seconds']), row['measured_steps'])
            median = np.median(row['times_seconds'])
            checks.close(prefix+'/median_step_ms', median*1000, row['median_step_ms'])
            checks.close(prefix+'/images_per_second', row['batch_size']/median, row['images_per_second'])
        raw_inference = read(art / 'benchmarks/inference.json')
        for name in CONFIG['architectures']:
            bench = stats['benchmark'][name]
            rounds = {}
            for amp, precision in [(False,'fp32'),(True,'amp')]:
                runs = sorted([r for r in raw_inference if r['architecture']==name and r['batch_size']==1 and r['amp']==amp],key=lambda r:r['round'])
                checks.close(f'{session}/{name}/{precision}/rounds', len(runs), 3)
                rounds[precision] = np.array([r['median_ms'] for r in runs])
                checks.close(f'{session}/{name}/{precision}/summary_median', np.median(rounds[precision]), bench[precision]['median_ms'])
            speedups = rounds['fp32']/rounds['amp']
            checks.close(f'{session}/{name}/precision_speedup', np.median(speedups), bench['speedup_median'])
            checks.close(f'{session}/{name}/precision_range', [min(speedups),max(speedups)], bench['speedup_round_range'])
            fit = [r for r in raw_inference if r['architecture']==name and r['batch_size']>1 and 'error' not in r]
            best = max(fit,key=lambda r:r['wall_images_per_second'])
            checks.close(f'{session}/{name}/best_throughput_batch', best['batch_size'], bench['best_probed_amp_batch'])
            checks.close(f'{session}/{name}/best_throughput', best['wall_images_per_second'], bench['best_probed_amp_images_per_second'])
            fp = next(r for r in training if r['architecture']==name and r['batch_size']==16 and not r['amp'])
            amp = next(r for r in training if r['architecture']==name and r['batch_size']==16 and r['amp'])
            successful = [r for r in training if r['architecture']==name and 'error' not in r]
            hardware_rows.append({'session':session,'architecture':name,
                                  'batch1_fp32_ms':bench['fp32']['median_ms'], 'batch1_amp_ms':bench['amp']['median_ms'],
                                  'batch1_amp_speedup':bench['speedup_median'],
                                  'synthetic_batch16_amp_speedup': fp['median_step_ms']/amp['median_step_ms'],
                                  'synthetic_batch16_allocation_reduction_percent':100*(1-amp['peak_allocated_mib']/fp['peak_allocated_mib']),
                                  'largest_tested_successful_training_batch':max(r['batch_size'] for r in successful),
                                  'amp_batch64_peak_mib':next(r['peak_allocated_mib'] for r in successful if r['batch_size']==64 and r['amp']),
                                  'description':'One session of short microbenchmarks, not an absolute hardware ceiling or an end-to-end acceleration guarantee.'})
        result['sessions'][session] = {'status':'PASS','runs':12,'test_images':624,
                                       'description':'Prediction metrics, seed aggregation, paired intervals and saved timing summaries checked against underlying evidence.'}
    csv_out(OUT/'tables/accepted_comparison.csv',accepted_summary)
    csv_out(OUT/'tables/precision_consistency.csv',precision_rows)
    csv_out(OUT/'tables/hardware_sessions.csv',hardware_rows)
    csv_out(OUT/'statistics/phase_5_checks.csv',checks.rows)
    trace = [
        ('test_curves.png','predictions/*seed42.npz','ROC and PR computed from real labels/probabilities; illustrative seed 42, no ensemble'),
        ('confusion_matrices.png','predictions/*seed42.npz','Counts from probability >=0.5; illustrative seed 42'),
        ('training_curves.png','history/*.json','Mean and sample SD at each epoch across three seeds'),
        ('performance_cost.png','statistical_summary.json + history/*_manifest.json','Per-model AUROC mean/SD, measured AMP forward latency and training wall duration'),
        ('hardware_scaling.png','benchmarks/inference.json','Recorded per-batch synchronized wall throughput and allocation'),
    ]
    csv_out(OUT/'tables/figure_provenance.csv',[{'figure':f,'source_relative_to_fresh_artifacts':s,'description':d} for f,s,d in trace])
    result.update({'checks_passed':len(checks.rows),'bootstrap':{'replicates':2000,'clusters':621,'seed':BOOTSTRAP_SEED,
                   'description':'Independent weighted positive-negative pair counting with half-credit for ties reproduces the original rank-sum resampling results within 1e-14.'},
                   'throughput_limitation':'Inference wall-throughput rates are recorded measurements. Original wall-duration denominators were not retained separately, so CUDA-event samples cannot independently reconstruct those denominators; case-selection/aggregation was checked against recorded rates. Synthetic step throughput is fully recomputed from saved synchronized wall step times.',
                   'primary_session':'retrained_v1','acceptance_scope':'Technical statistical/artifact review; not supervisor or clinical approval.'})
    save(OUT/'statistics/phase_5_acceptance.json',result)
    note = f'''# Phase 5 — evaluation and statistics accepted

{len(checks.rows)} numerical checks passed across both sessions. Primary report session: `retrained_v1`; the pilot regeneration remains separate reproduction evidence.

An independent implementation rebuilt all 2,000 paired bootstrap draws using integer case multiplicities and positive-negative pair counting with half-credit for tied probabilities, rather than calling the original rank-sum bootstrap. The draws matched within absolute tolerance 1e-14. Means, sample SD (ddof=1), interval endpoints, per-seed confusion counts and precision discrepancies were recomputed from real prediction arrays.

Accepted tables are in `06_results/tables/`; detailed checks and independent draws are in `06_results/statistics/`. Figure sources are mapped in `figure_provenance.csv`. Original and fresh predictive results agree; hardware measurements remain separate sessions.

The primary endpoint is AUROC. Other metrics use threshold 0.5. Three models are compared using three seeds each; frozen ResNet is an extra paired control. Aggregate scores are means of per-model scores, not a probability ensemble. SD across three seeds and the exploratory seed-and-image-cluster bootstrap represent different variability summaries. Unknown patient dependence remains unmeasured; intervals are not multiplicity adjusted.

The ResNet fine-tuning interval is above zero within this benchmark. Both other declared architecture-difference intervals include zero, so close means do not establish ordering or equivalence. No literature scores are mixed into local result tables.

Raw CUDA-event medians, means, p95 and interleaved precision summaries were checked. Synthetic synchronized training-step medians and throughput were independently recomputed. Inference wall-throughput denominators were not separately saved; those rates can be traced to raw measurement records and code, while aggregation is verified against the recorded rates. Event sums must not be substituted for synchronized wall duration.

Five original result figures have code/source mappings. They use seed 42 for representative test curves/confusion matrices and all seeds for training/summary plots. No extra GPU experiment was needed for acceptance.
'''
    (OUT/'PHASE_5_STATISTICAL_REVIEW.md').write_text(note,encoding='utf-8')
    print(f'PHASE 5 PASS: {len(checks.rows)} numerical checks',flush=True)
    return result, hardware_rows, precision_rows


if __name__ == '__main__':
    review()