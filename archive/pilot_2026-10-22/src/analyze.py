"""Generate statistics, plots, and the report from saved experiment outputs only."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
import csv
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ART = ROOT / 'artifacts'
FIG = ART / 'figures'
GROUPS = ['resnet18_frozen', 'resnet18_partial', 'densenet121_partial', 'mobilenet_v3_small_partial']
LABELS = ['ResNet-18 frozen', 'ResNet-18 partial FT', 'DenseNet-121 partial FT', 'MobileNetV3-Small partial FT']
COLORS = ['#64748b', '#2563eb', '#dc2626', '#059669']
SEEDS = [42, 43, 44]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def auc_rows(labels, predictions):
    ranks = rankdata(predictions, axis=-1)
    pos = labels == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    return (ranks[..., pos].sum(axis=-1) - npos * (npos + 1) / 2) / (npos * nneg)


def bootstrap(predictions, y, groups):
    rng = np.random.default_rng(20261022)
    out = np.empty((2000, len(GROUPS)))
    for i in range(len(out)):
        selected = rng.integers(0, len(groups), len(groups))
        ix = np.concatenate([groups[j] for j in selected])
        seed_ix = rng.integers(0, len(SEEDS), len(SEEDS))
        out[i] = auc_rows(y[ix], predictions[:, seed_ix][:, :, ix]).mean(axis=1)
    return out


def finish_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_report():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    rows = load_json(ART / 'test_metrics.json')
    benches = load_json(ART / 'benchmarks' / 'inference.json')
    train_capacity = load_json(ART / 'benchmarks' / 'training_capacity.json')
    audit = load_json(ART / 'dataset_audit.json')
    env = load_json(ART / 'environment.json')
    config = load_json(ROOT / 'configs' / 'experiment.json')
    manifests, histories = {}, {}
    p = np.empty((4, 3, 624))
    fp32 = np.empty_like(p)
    for i, group in enumerate(GROUPS):
        for j, seed in enumerate(SEEDS):
            rid = f'{group}_seed{seed}'
            data = np.load(ART / 'predictions' / f'{rid}.npz')
            if i == 0 and j == 0:
                y = data['labels'].astype(int)
            assert np.array_equal(y, data['labels'])
            p[i, j] = data['probabilities']
            fp32[i, j] = data['fp32_probabilities']
            manifests[rid] = load_json(ART / 'history' / f'{rid}_manifest.json')
            histories[rid] = load_json(ART / 'history' / f'{rid}.json')
            saved = next(r for r in rows if r['id'] == rid)
            assert abs(roc_auc_score(y, p[i, j]) - saved['auroc']) < 1e-12
    data = np.load(ROOT / 'data' / 'pneumoniamnist_224.npz')
    test_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in data['test_images']]
    unique_hashes = list(dict.fromkeys(test_hashes))
    clusters = [np.flatnonzero(np.array(test_hashes) == h) for h in unique_hashes]
    ci_samples = bootstrap(p, y, clusters)
    np.savez(ART / 'bootstrap_samples.npz', mean_auroc=ci_samples)
    summary = []
    metric_names = ['auroc', 'ap', 'f1', 'precision', 'recall', 'specificity', 'accuracy', 'balanced_accuracy']
    for i, group in enumerate(GROUPS):
        group_rows = [r for r in rows if r['id'].startswith(group + '_seed')]
        item = {'group': group, 'label': LABELS[i], 'seeds': SEEDS,
                'auroc_ci95': np.percentile(ci_samples[:, i], [2.5, 97.5]).tolist()}
        for metric in metric_names:
            vals = [r[metric] for r in group_rows]
            item[metric + '_mean'] = float(np.mean(vals))
            item[metric + '_sd'] = float(np.std(vals, ddof=1))
        ms = [manifests[f'{group}_seed{s}'] for s in SEEDS]
        item['parameters'] = ms[0]['parameters']
        item['weights_mib'] = float(np.mean([m['checkpoint_bytes_fp32'] / 2**20 for m in ms]))
        item['training_minutes_mean'] = float(np.mean([m['total_seconds_including_validation'] / 60 for m in ms]))
        item['training_peak_mib'] = max(d['peak_allocated_mib_including_data'] for m in ms for d in m['phase_details'])
        item['partial_trainable_parameters'] = histories[f'{group}_seed42'][-1]['trainable_parameters']
        summary.append(item)
    pairs = []
    for a, b in [(1, 0), (1, 2), (3, 2)]:
        dif = ci_samples[:, a] - ci_samples[:, b]
        pairs.append({'comparison': f'{LABELS[a]} minus {LABELS[b]}',
                      'delta_auroc_mean': summary[a]['auroc_mean'] - summary[b]['auroc_mean'],
                      'ci95': np.percentile(dif, [2.5, 97.5]).tolist(),
                      'seed_deltas': (auc_rows(y, p[a]) - auc_rows(y, p[b])).tolist()})
    bench_summary = {}
    for name in config['architectures']:
        modes = {}
        for amp in [False, True]:
            rounds = sorted([r for r in benches if r.get('batch_size') == 1 and r.get('architecture') == name and r.get('amp') == amp], key=lambda r: r['round'])
            modes['amp' if amp else 'fp32'] = {
                'median_ms': float(np.median([r['median_ms'] for r in rounds])),
                'p95_ms': float(np.median([r['p95_ms'] for r in rounds])),
                'round_medians_ms': [r['median_ms'] for r in rounds],
                'peak_mib': max(r['peak_allocated_mib'] for r in rounds)}
        speedups = np.array(modes['fp32']['round_medians_ms']) / np.array(modes['amp']['round_medians_ms'])
        modes['speedup_median'] = float(np.median(speedups))
        modes['speedup_round_range'] = [float(speedups.min()), float(speedups.max())]
        batches = [r for r in benches if r.get('architecture') == name and r.get('batch_size', 0) > 1 and 'error' not in r]
        best = max(batches, key=lambda r: r['wall_images_per_second'])
        modes['best_probed_amp_batch'] = best['batch_size']
        modes['best_probed_amp_images_per_second'] = best['wall_images_per_second']
        modes['best_probed_peak_mib'] = best['peak_allocated_mib']
        modes['largest_successfully_probed_batch'] = max(r['batch_size'] for r in batches)
        bench_summary[name] = modes
    stats = {'summary': summary, 'paired_comparisons': pairs,
             'bootstrap': {'repetitions': 2000, 'test_exact_image_clusters': len(clusters),
                           'seed': 20261022, 'method': 'paired training-seed and exact-image-cluster percentile bootstrap; no patient IDs'},
             'benchmark': bench_summary, 'synthetic_training_capacity': train_capacity}
    save_json(ART / 'statistical_summary.json', stats)
    with (ART / 'comparison_summary.csv').open('w', newline='') as f:
        scalar_keys = [k for k, v in summary[0].items() if not isinstance(v, list)]
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in scalar_keys} for row in summary])

    # Three seeds are individual models; curves and confusion matrices use seed42,
    # not an unreported probability ensemble.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y, p[i, 0])
        precision, recall, _ = precision_recall_curve(y, p[i, 0])
        axes[0].plot(fpr, tpr, color=COLORS[i], label=f'{label}: {roc_auc_score(y,p[i,0]):.3f}')
        axes[1].plot(recall, precision, color=COLORS[i], label=label)
    axes[0].plot([0, 1], [0, 1], '--', color='#94a3b8')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC — seed 42')
    axes[1].axhline(y.mean(), ls='--', color='#94a3b8', label=f'Prevalence {y.mean():.3f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Test precision–recall — seed 42')
    for ax in axes:
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(alpha=.15)
    finish_figure(fig, 'test_curves.png')

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for i, ax in enumerate(axes):
        cm = confusion_matrix(y, p[i, 0] >= .5)
        ax.imshow(cm, cmap='Blues')
        for (r, c), value in np.ndenumerate(cm):
            ax.text(c, r, str(value), ha='center', va='center', color='white' if value > cm.max()/2 else '#0f172a')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Pneumonia'],
               yticklabels=['Normal', 'Pneumonia'], xlabel='Predicted', title=LABELS[i].replace(' partial FT', '\npartial FT'))
    axes[0].set_ylabel('True label')
    fig.suptitle('Test confusion matrices — seed 42, fixed threshold 0.5')
    finish_figure(fig, 'confusion_matrices.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, group in enumerate(GROUPS):
        for ax, key in zip(axes, ['train_loss', 'val_auroc']):
            values = np.array([[r[key] for r in histories[f'{group}_seed{s}']] for s in SEEDS])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(np.arange(1, 6), mean, color=COLORS[i], label=LABELS[i])
            ax.fill_between(np.arange(1, 6), mean-sd, mean+sd, alpha=.1, color=COLORS[i])
            ax.set_xlabel('Epoch')
            ax.axvline(2.5, color='#94a3b8', ls=':', lw=1)
            ax.grid(alpha=.15)
    axes[0].set(title='Training weighted BCE — mean ± seed SD', ylabel='Weighted BCE')
    axes[1].set(title='Validation AUROC — mean ± seed SD', ylabel='AUROC')
    axes[1].legend(fontsize=7)
    finish_figure(fig, 'training_curves.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for i in [1, 2, 3]:
        s = summary[i]
        b = bench_summary[GROUPS[i].removesuffix('_partial')]
        axes[0].errorbar(b['amp']['median_ms'], s['auroc_mean'], yerr=s['auroc_sd'], fmt='o', color=COLORS[i], capsize=4)
        axes[0].annotate(LABELS[i].replace(' partial FT',''), (b['amp']['median_ms'], s['auroc_mean']), xytext=(7, 4), textcoords='offset points', fontsize=8)
    axes[0].set(xlabel='Batch-1 AMP forward latency (ms)', ylabel='Test AUROC (mean ± seed SD)', title='Observed performance–latency trade-off')
    axes[0].margins(x=.35, y=.25)
    axes[1].barh(LABELS, [s['training_minutes_mean'] for s in summary], color=COLORS)
    axes[1].set(xlabel='Mean minutes per 5-epoch configuration', title='Training cost including validation')
    finish_figure(fig, 'performance_cost.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for i, name in enumerate(config['architectures']):
        bs = sorted([r for r in benches if r['architecture'] == name and r['batch_size'] > 1 and 'error' not in r], key=lambda r: r['batch_size'])
        axes[0].plot([r['batch_size'] for r in bs], [r['wall_images_per_second'] for r in bs], 'o-', label=name)
        axes[1].plot([r['batch_size'] for r in bs], [r['peak_allocated_mib'] for r in bs], 'o-', label=name)
    axes[0].set(xlabel='Inference batch size', ylabel='Images / second', title='AMP throughput — tested batch sizes only')
    axes[1].axhline(env['gpu_vram_mib'], ls='--', color='#dc2626', label='Physical GPU VRAM')
    axes[1].set(xlabel='Inference batch size', ylabel='Peak allocated MiB', title='Model/input/activation allocation')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    finish_figure(fig, 'hardware_scaling.png')

    names = [s['label'] for s in summary]
    metric_table = '\n'.join(
        f"| {s['label']} | {s['auroc_mean']:.4f} ± {s['auroc_sd']:.4f} | {s['auroc_ci95'][0]:.4f}–{s['auroc_ci95'][1]:.4f} | {s['ap_mean']:.4f} | {s['f1_mean']:.4f} | {s['recall_mean']:.4f} | {s['specificity_mean']:.4f} | {s['accuracy_mean']:.4f} |"
        for s in summary)
    resource_table = '\n'.join(
        f"| {s['label']} | {s['parameters']:,} | {s['partial_trainable_parameters']:,} | {s['weights_mib']:.2f} | {s['training_minutes_mean']:.2f} | {s['training_peak_mib']:.1f} |"
        for s in summary)
    delta_table = '\n'.join(f"| {r['comparison']} | {r['delta_auroc_mean']:+.4f} | {r['ci95'][0]:+.4f} to {r['ci95'][1]:+.4f} | {', '.join(f'{v:+.4f}' for v in r['seed_deltas'])} |" for r in pairs)
    benchmark_table = '\n'.join(f"| {name} | {b['fp32']['median_ms']:.3f} | {b['amp']['median_ms']:.3f} | {b['speedup_median']:.2f}× | {b['speedup_round_range'][0]:.2f}–{b['speedup_round_range'][1]:.2f}× | {b['fp32']['peak_mib']:.1f} / {b['amp']['peak_mib']:.1f} |" for name, b in bench_summary.items())
    batch_table = '\n'.join(f"| {name} | {b['best_probed_amp_batch']} | {b['best_probed_amp_images_per_second']:.1f} | {b['best_probed_peak_mib']:.1f} | {b['largest_successfully_probed_batch']} |" for name, b in bench_summary.items())
    training_capacity_table = '\n'.join(
        f"| {r['architecture']} | {r['batch_size']} | {'AMP' if r['amp'] else 'FP32'} | {r['median_step_ms']:.2f} | {r['images_per_second']:.1f} | {r['peak_allocated_mib']:.1f} |"
        if 'error' not in r else f"| {r['architecture']} | {r['batch_size']} | {'AMP' if r['amp'] else 'FP32'} | OOM | — | — |"
        for r in train_capacity)
    training_precision_table = []
    for name in config['architectures']:
        base = next(r for r in train_capacity if r['architecture'] == name and r['batch_size'] == 16 and not r['amp'])
        mixed = next(r for r in train_capacity if r['architecture'] == name and r['batch_size'] == 16 and r['amp'])
        if 'error' not in base and 'error' not in mixed:
            speed = base['median_step_ms'] / mixed['median_step_ms']
            reduction = 100 * (1 - mixed['peak_allocated_mib'] / base['peak_allocated_mib'])
            training_precision_table.append(f"| {name} | {speed:.2f}× | {reduction:.1f}% |")
    training_precision_table = '\n'.join(training_precision_table)
    seed_table = '\n'.join(f"| {r['id']} | {r['auroc']:.4f} | {r['ap']:.4f} | {r['f1']:.4f} | {r['recall']:.4f} | {r['specificity']:.4f} |" for r in rows)
    amp_table = '\n'.join(f"| {s['label']} | {max(r['amp_fp32_max_probability_difference'] for r in rows if r['id'].startswith(s['group']+'_seed')):.6f} | {sum(r['amp_fp32_decision_disagreements'] for r in rows if r['id'].startswith(s['group']+'_seed'))} / {3*len(y)} | {max(abs(r['auroc']-r['fp32_auroc']) for r in rows if r['id'].startswith(s['group']+'_seed')):.6f} |" for s in summary)
    winner = max(summary[1:], key=lambda s: s['auroc_mean'])
    f1_winner = max(summary[1:], key=lambda s: s['f1_mean'])
    compact = summary[3]
    d = pairs[0]
    optimization_verdict = ('The interval is entirely above zero, supporting an improvement within this benchmark and protocol.' if d['ci95'][0] > 0 else
                            'The interval crosses zero; these results do not establish a reliable AUROC improvement.' if d['ci95'][1] >= 0 else
                            'The interval is entirely below zero; this fine-tuning recipe worsened AUROC in the measured comparison.')
    mobile_delta = compact['auroc_mean'] - winner['auroc_mean']
    training_wall = sum(m['total_seconds_including_validation'] for m in manifests.values())
    # Shared ResNet warm-up phases appear in both branch manifests; subtract once per seed.
    actual_train_seconds = training_wall - sum(manifests[f'resnet18_frozen_seed{s}']['phase_details'][0]['seconds_including_validation'] for s in SEEDS)
    now = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %B %Y, %H:%M IST')
    report = f'''# Comparative analysis of pretrained CNNs for pediatric pneumonia classification

Completed: {now}. Hardware: {env['gpu']}, {env['gpu_vram_mib']:.1f} MiB VRAM. This report contains measured results from this workspace.

## Abstract

We compared ImageNet-pretrained ResNet-18, DenseNet-121, and MobileNetV3-Small on the 224-resolution PneumoniaMNIST+ pediatric pneumonia-versus-normal benchmark. Each architecture used a fixed two-epoch head-training phase followed by three epochs of partial fine-tuning, repeated with seeds 42, 43, and 44. A paired ResNet experiment continued head-only training from the same initial checkpoint. The highest mean test AUROC among the three partially fine-tuned models was {winner['auroc_mean']:.4f}, achieved by {winner['label']}. ResNet partial fine-tuning changed mean AUROC by {d['delta_auroc_mean']:+.4f} relative to its frozen-feature control, with an exploratory paired 95% interval of [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]. Training, latency, memory, and precision measurements quantify the observed cost trade-offs on a 6 GB laptop GPU. The findings are limited to this processed pediatric dataset and short training budget.

## 1. Question and contribution

The project asks which of three established pretrained CNN packages provides the most useful observed predictive-performance and resource trade-off under a fixed short training protocol. It also asks whether partial fine-tuning improves a frozen ResNet baseline and whether mixed precision improves inference efficiency on the available GPU.

The contribution is an educational empirical comparison with saved data provenance, exact-duplicate cleanup, three training seeds, paired uncertainty estimates, measured device costs, and a reproducible pipeline. No new architecture or clinical-performance claim is made.

## 2. Related work

ChestX-ray8 introduced a large report-labeled chest X-ray dataset; its original counts and the later ChestX-ray14 counts refer to different releases. CheXNet established DenseNet-121 as an influential pneumonia baseline and reported a specific radiologist comparison rather than universal clinical superiority. CheXpert introduced uncertainty-aware report labels and expert evaluation for selected findings. [Wang et al.](https://arxiv.org/abs/1705.02315), [CheXNet](https://arxiv.org/abs/1711.05225), [CheXpert](https://arxiv.org/abs/1901.07031)

Bressem et al. and CheXtransfer motivate controlled architecture and efficiency comparisons. Their tasks, datasets, and budgets differ from ours; their published scores are not directly comparable with our table. Our partial fine-tuning experiment is not a reproduction of CheXtransfer's truncation experiment. [Bressem et al.](https://www.nature.com/articles/s41598-020-70479-z), [CheXtransfer](https://arxiv.org/abs/2101.06871)

## 3. Dataset, scope, and cleanup

We used the actual 224 × 224 PneumoniaMNIST+ file, not an upscaled 28-pixel version. The file was downloaded from the official MedMNIST+ release and its MD5 matched `{audit['md5']}`. It contains pediatric normal/pneumonia images, not adult multi-finding classification. [Official release](https://zenodo.org/records/10519652), [dataset construction](https://www.nature.com/articles/s41597-022-01721-8)

The original splits contained 4,708 training, 524 validation, and 624 test images. Exact SHA-256 image matching found eight unique image hashes shared by training and validation, affecting ten training images. These ten training examples were excluded. No exact train–test or validation–test overlap was found. The working split sizes were **4,698 / 524 / 624**. Within-split duplicate excess was 18 training images before cleanup and 3 test images; the test split was preserved for comparability. Patient identifiers are absent from the NPZ, so patient independence could not be independently verified. Exact-image hashes do not detect all repeat patients or near duplicates.

The test set contains 234 normal and 390 pneumonia labels. A constant majority-class prediction has accuracy 0.625, AUROC 0.5, and AP equal to prevalence (0.625). These references show why accuracy alone is inadequate.

## 4. Protocol

Inputs were grayscale images repeated into three channels, scaled to [0,1] and normalized with ImageNet channel statistics. Training alone used random rotations within ±5°, bilinear sampling, and border padding. There were no flips or random crops. Image arrays were cached on the GPU as uint8; conversion, rotation, and normalization occurred per batch.

All models used ImageNet-1K V1 weights from torchvision {env['torchvision']}, a one-logit output, batch size 16, weighted BCEWithLogitsLoss with positive weight computed from training labels, and AdamW with weight decay 1e-4. Epochs 1–2 trained the head at learning rate 1e-3. Epochs 3–5 used learning rate 1e-4. The final ResNet residual stage, final DenseNet dense block/normalization, or last two MobileNet feature modules were unfrozen. Batch-normalization running statistics stayed fixed. CUDA FP16 autocast and gradient scaling with initial scale 1024 were used throughout training. TF32 was disabled. Fixed final checkpoints were evaluated; there was no test-dependent checkpoint selection or threshold tuning. Validation/test predictions used evaluation mode and batches of 64, with a smaller final batch where necessary; batch-1 timings are a separate resource measurement.

The ResNet frozen and partial branches started from an identical epoch-2 checkpoint, used the same phase-2 rate, three additional epochs, seed schedule, batch order, and augmentation draws. Fresh optimizers were used for both branches. The saved checkpoint hashes and manifests record the common starting state. This controls training exposure for the fine-tuning comparison.

Each configuration was trained with seeds 42, 43, and 44. The three architecture comparisons use partial fine-tuning. Frozen ResNet is an extra control. Heads and final stages differ in trainable capacity, so this is a comparison of pretrained packages under the chosen recipe rather than a causal isolation of architecture. Training cost includes validation and excludes weight download/setup. Reusing the common ResNet head phase, the actual formal-training duration was approximately **{actual_train_seconds/60:.1f} minutes**.

## 5. Comparative predictive results

AUROC is the primary endpoint. Values below are means across three independently seeded trained checkpoints; ± denotes sample standard deviation across seeds. Other metrics use a fixed 0.5 threshold. AP is average precision, not trapezoidal PR-AUC.

| Configuration | AUROC mean ± seed SD | Exploratory 95% AUROC interval | AP | F1 | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
{metric_table}

The highest observed mean AUROC among partially fine-tuned models was **{winner['label']}**. MobileNetV3-Small's mean AUROC difference from that model was **{mobile_delta:+.4f}**. These are measurements under this protocol, not universal rankings.

The highest mean F1 at threshold 0.5 was **{f1_winner['label']}**, at {f1_winner['f1_mean']:.4f} ± {f1_winner['f1_sd']:.4f} across seeds. ResNet partial fine-tuning had F1 {summary[1]['f1_mean']:.4f} ± {summary[1]['f1_sd']:.4f}. AUROC measures ranking across thresholds; it does not guarantee the best fixed-threshold operating point. The per-seed table exposes the sensitivity/specificity variation rather than hiding it in a single average.

![Test ROC and precision–recall curves](artifacts/figures/test_curves.png)

Curves below and confusion matrices use seed 42 as a predefined representative run. They are not ensembles. Aggregate metrics above average per-seed scores, not probabilities.

![Confusion matrices](artifacts/figures/confusion_matrices.png)

### Uncertainty and paired differences

We used 2,000 paired hierarchical bootstrap replicates with a fixed RNG seed. Each replicate resampled the three training seeds and the 621 exact-image-hash clusters in the test set, preserving repeated identical-image membership. The same resampled seeds and cases were used for both models in each difference. Intervals are percentile intervals for mean AUROC. They remain exploratory with only three seeds and unknown patient clustering; they are not multiplicity-adjusted confirmatory tests.

| Comparison | Mean AUROC difference | Exploratory 95% interval | Differences for seeds 42, 43, 44 |
|---|---|---|---|
{delta_table}

The ResNet-versus-DenseNet and MobileNet-versus-DenseNet intervals both include zero. Their small observed AUROC differences do not establish a reliable architecture ordering, and a nonsignificant-looking difference is not proof of equivalence. The stronger signal here is the paired frozen-versus-partial ResNet comparison.

## 6. Resource comparison

Weights are FP32 state dictionaries without optimizer state. Peak training allocation includes the resident image arrays, model state, optimizer, activations, and any live tensors in that phase; it is not total system GPU usage or peak reserved memory. Trainable parameters refer to the final phase.

| Configuration | Total parameters | Final-phase trainable parameters | Weights MiB | Mean training minutes | Peak allocated MiB |
|---|---|---|---|---|---|
{resource_table}

![Performance and training cost](artifacts/figures/performance_cost.png)

![Training and validation curves](artifacts/figures/training_curves.png)

Training loss is class-weighted BCE. Validation AUROC measures ranking. These curves describe a five-epoch experiment and do not establish convergence.

## 7. Optimization proof of concept

### Partial fine-tuning

For ResNet, the mean change was **{d['delta_auroc_mean']:+.4f} AUROC**, interval **[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]**. {optimization_verdict} The resource table shows the extra trainable capacity and measured training cost. Fine-tuning changes the weights learned; it does not compress the deployed architecture or reduce its parameter count.

### Mixed-precision inference

The same seed-42 checkpoints were benchmarked on device-resident batch-1 inputs in FP32 and FP16 autocast. We ran three precision-interleaved rounds, each with 30 warm-up and 200 timed forwards, using CUDA events and synchronization. Table latency is the median of round medians; speedup is the median of paired round ratios. The range shows the three round ratios. These are technical repeats on one laptop session, not independent hardware experiments. Timing excludes image decoding, preprocessing, transfer, and application overhead.

| Architecture | FP32 median ms | AMP median ms | Median speedup | Speedup round range | FP32 / AMP peak MiB |
|---|---|---|---|---|---|
{benchmark_table}

Precision consistency was evaluated across all three seeds per configuration on the complete test set. A small discrepancy is expected from reduced precision; it should be measured rather than assumed absent.

| Configuration | Largest probability difference | Threshold disagreements across seed predictions | Largest absolute AUROC change |
|---|---|---|
{amp_table}

These precision results must be read together with the timing results. A speedup below 1× is a slowdown. Fewer bits did not automatically make an isolated batch-1 request faster, and any probability or decision discrepancies mean reduced precision cannot be described as numerically identical to FP32. In particular, MobileNet's measured maximum probability discrepancy should be reviewed before treating AMP as an accuracy-preserving deployment setting. No test-driven retraining or threshold adjustment was performed to hide these discrepancies.

### Measured hardware envelope

AMP inference batches 16, 32, 64, and 128 were probed, subject to a 70%-of-VRAM allocated-memory stop rule. The table reports the best measured throughput and the largest successful tested batch. It does **not** establish the maximum possible batch size or a theoretical optimization ceiling. Peak allocation excludes driver/desktop allocations and framework-reserved-but-unused memory.

| Architecture | Best measured batch | Forward images/s | Peak allocated MiB at that batch | Largest successful tested batch |
|---|---|---|---|---|
{batch_table}

![Throughput and GPU allocation](artifacts/figures/hardware_scaling.png)

The GPU's physical capacity is approximately 6 GB. These experiments show which measured workloads fit and what their observed costs are. They do not measure full-backbone training at arbitrary resolutions, larger architectures, or the limits of every compression approach. All runs used batch 16; the inference batch probe cannot be used as a claim about training batch capacity.

### Synthetic full-gradient training capacity

To separate inference capacity from training capacity, we also loaded copies of seed-42 checkpoints and measured full-gradient AdamW steps on synthetic 224-pixel inputs. All model parameters were enabled, BN running statistics stayed fixed, and no preprocessing/transfer was timed. Each setting used three warm-up and ten measured steps. Updated in-memory copies were discarded; the comparative checkpoints and their evaluation results were not changed. These are microbenchmarks, not additional predictive training experiments.

| Architecture | Batch | Precision | Median full step ms | Images/s | Peak allocated MiB |
|---|---|---|---|---|---|
{training_capacity_table}

For the same batch size of 16, the following ratios compare the synthetic full-gradient step measurements. The short probe has no independent run-to-run confidence interval; use it as an engineering estimate, not a guaranteed end-to-end training acceleration.

| Architecture | AMP step speedup versus FP32 | Peak allocation reduction |
|---|---|---|
{training_precision_table}

Successful training probes show that the tested configuration fits; they do not establish the maximum stable batch under long training, and no exact capacity limit is inferred unless a measured failure is recorded. Peak allocated memory is a lower bound on total GPU memory requirements because driver and reserved memory are additional.

## 8. What the evidence supports next

Keep the completed comparison as the college project's core. Treat the paired ResNet result and the precision measurements as initial optimization evidence. Use the measured trade-off rather than declaring the smallest or deepest network best in advance.

**Practical recommendation:** continue with ResNet-18 as the main accuracy/latency baseline and keep MobileNetV3-Small as the storage-efficient alternative. ResNet had the highest observed mean AUROC and lowest batch-1 forward latency; MobileNet had the highest mean fixed-threshold F1 and used {100*(1-summary[3]['parameters']/summary[1]['parameters']):.1f}% fewer parameters than ResNet. This recommendation does not assert that ResNet's AUROC is statistically superior to DenseNet. For isolated batch-1 predictions on this setup, FP32 was faster than AMP. For full-gradient training, the synthetic probe supports AMP for ResNet and DenseNet, while MobileNet traded lower memory for slower batch-16 steps.

For the next hardware-constrained experiment, ResNet batch 32 or 64 and DenseNet batch 32 are reasonable *tested starting settings* at 224 pixels. DenseNet's full-gradient AMP batch 64 allocated approximately 4.17 GiB before additional driver/reserved allocations, so its remaining practical headroom is smaller. Increasing batch size changes optimization behavior and requires a new validation-controlled experiment; the microbenchmark alone does not prove equal predictive performance.

The next controlled extension is a longer training budget for the selected model, using validation data for any development choices and a new untouched evaluation set for a new confirmatory claim. A later compression study can test truncation or distillation, one mechanism at a time, measuring predictive loss and actual latency. Quantization must be evaluated on its intended runtime; fewer bits alone do not imply an RTX GPU speedup. Cross-hospital adult findings require a separately scoped dataset and cannot be inferred from pediatric pneumonia classification.

## 9. Limitations

This is a single pediatric binary benchmark, processed to 224 pixels, without external validation. Patient IDs were unavailable, and exact-duplicate checks are incomplete leakage checks. Three seeds provide only limited information about training variability. The short fixed schedule and shared hyperparameters may favor some models; no architecture-specific tuning or convergence claim is made. ImageNet V1 pretraining recipes differ. Weighted-loss probabilities are not claimed to be calibrated. The test set was preserved with three duplicate excess images, handled as exact-image clusters in the bootstrap. Confidence intervals do not account for unknown patient dependence. Device timing is hardware-, software-, precision-, and session-specific. The benchmark creators specify educational/research use rather than clinical deployment. [MedMNIST](https://medmnist.com/)

## 10. Per-seed results and reproducibility

| Run | AUROC | AP | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
{seed_table}

The workspace includes the configuration, protocol timestamp/source hash, dataset audit, smoke-test results, epoch histories, checkpoint hashes, saved test probabilities in both precisions, bootstrap draws, raw timing samples, CSV summaries, figures, and locked library versions. See `README.md` for commands. Source files and saved numerical outputs, rather than this prose, are the record of the experiment.

Environment: Python 3.13.7, PyTorch {env['torch']}, torchvision {env['torchvision']}, CUDA runtime {env['cuda_runtime']}, Windows, four PyTorch CPU threads. Pretrained weights were official ImageNet-1K V1 packages. PyTorch documents the AMP APIs and the compatible package pair. [AMP documentation](https://docs.pytorch.org/docs/2.10/amp.html), [version pairing](https://pytorch.org/get-started/previous-versions/)

Original source-data attribution: Kermany et al. (2018), *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, DOI 10.1016/j.cell.2018.02.010, as identified in the MedMNIST dataset-construction reference. [Original source paper](https://doi.org/10.1016/j.cell.2018.02.010)
'''
    report = report.replace('](artifacts/figures/', f']({FIG.as_posix()}/')
    (ROOT / 'comparative_analysis_report.md').write_text(report, encoding='utf-8')
    save_json(ART / 'analysis_manifest.json', {'generated_ist': now,
              'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runs': len(rows), 'test_cases': len(y), 'exact_image_clusters': len(clusters),
              'training_minutes_actual_excluding_setup': actual_train_seconds / 60,
              'predictions_recomputed_metrics_match': True})
    print(json.dumps(stats, indent=2))
    print('REPORT AND FIGURES COMPLETE')


if __name__ == '__main__':
    make_report()
