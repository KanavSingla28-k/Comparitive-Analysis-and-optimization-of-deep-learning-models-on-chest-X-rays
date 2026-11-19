"""Validation-only ResNet batch-size pilot. Run with --help for reproduction commands.

Uses no test arrays. Python constants define the protocol; CSV/Markdown are the
primary review outputs and explained JSON is a machine-readable companion.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = PROJECT / '05_experiments' / 'batch32_v2_pilot'
PROTOCOL = {
    'id': 'cxr_batch32_v2_pilot', 'architecture': 'resnet18',
    'seeds': [42, 43, 44], 'batches': [16, 32], 'sessions': [1, 2],
    'epochs': 3, 'start_epoch': 2, 'learning_rate': 1e-4,
    'weight_decay': 1e-4, 'rotation_degrees': 5, 'threshold': 0.5,
    'max_memory_fraction': .70, 'max_mean_auroc_drop': .005,
    'max_seed_auroc_drop': .010, 'minimum_median_speedup': 1.10,
    'timing': 'Adaptation training plus per-epoch validation; excludes model loading, warm-up and checkpoint writing.',
    'augmentation': 'One full-epoch angle vector generated before batching; identical per-image angles and image order for each paired seed/epoch.',
    'precision': 'CUDA float16 autocast and GradScaler; validation inference batch 64.',
    'selection': 'Validation only. Fixed final epoch; no test labels or predictions accessed.',
    'timing_repeats': 'Two technical sessions, reversed order. Three independent training seeds, not six.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gpu_status():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,power.draw,temperature.gpu', '--format=csv'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)


def verify(run):
    manifest = json.loads((run / 'integrity.json').read_text(encoding='utf-8'))
    failures = [name for name, digest in manifest.items() if not (run / name).is_file() or sha(run / name) != digest]
    if failures:
        raise RuntimeError(f'Integrity failures: {failures}')
    print(f'PASS: {len(manifest)} pilot artifact hashes verified.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--verify', action='store_true', help='Verify saved artifacts without training.')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(PROJECT / '05_experiments') or run == PROJECT / '05_experiments':
        raise ValueError('Run directory must be a child of project/05_experiments')
    if args.verify:
        verify(run)
        return
    if run.exists():
        raise FileExistsError('Use a new run directory; existing pilot results are never overwritten.')
    os.environ['IRS_RUN_DIR'] = str(run)
    # runtime sets offline dependencies and the active cache before scientific imports.
    import runtime
    import experiment as e
    import numpy as np
    import torch
    import torch.nn.functional as F
    from settings import DATASET_SHA256
    e.setup()
    for directory in ['source', 'checkpoints', 'predictions', 'history', 'figures']:
        (run / directory).mkdir(exist_ok=True)
    source_names = ['batch32_pilot.py', 'experiment.py', 'settings.py', 'runtime.py']
    for name in source_names:
        shutil.copy2(Path(__file__).parent / name, run / 'source' / name)
    shutil.copy2(PROJECT / '05_experiments' / 'NEXT_OPTIMIZATION_EXPERIMENT.md', run / 'source' / 'original_proposal.md')
    dump(run / 'protocol.json', PROTOCOL)
    assert sha(runtime.DATA_FILE) == DATASET_SHA256, 'Dataset hash mismatch'
    originals = PROJECT / '05_experiments' / 'retrained_v1' / 'artifacts' / 'checkpoints'
    original_hashes = {str(p.relative_to(PROJECT)): sha(p) for p in originals.glob('*.pt')}
    arrays = e.ImageArrays.__new__(e.ImageArrays)
    arrays.x, arrays.y, arrays.indices = {}, {}, {}
    inputs = {'dataset_sha256': DATASET_SHA256, 'original_checkpoint_sha256': original_hashes,
              'source_sha256': {name: sha(run / 'source' / name) for name in source_names},
              'index_sha256': {}}
    # Access only train/val NPZ members; do not call raw_data()/audit_data().
    with np.load(runtime.DATA_FILE) as data:
        for split in ['train', 'val']:
            index_path = PROJECT / '03_data' / 'manifests' / f'{split}_indices.npy'
            idx = np.load(index_path)
            inputs['index_sha256'][split] = sha(index_path)
            arrays.indices[split] = idx
            arrays.x[split] = torch.from_numpy(data[f'{split}_images'][idx]).to(e.DEVICE)
            arrays.y[split] = torch.from_numpy(data[f'{split}_labels'][idx].reshape(-1).astype(np.float32)).to(e.DEVICE)
    assert len(arrays.y['train']) == 4698 and len(arrays.y['val']) == 524
    arrays.mean = torch.tensor([.485, .456, .406], device=e.DEVICE)[None, :, None, None]
    arrays.std = torch.tensor([.229, .224, .225], device=e.DEVICE)[None, :, None, None]
    dump(run / 'inputs.json', inputs)
    environment = {'created_utc': e.stamp(), 'python': sys.version, 'torch': torch.__version__,
                   'torchvision': e.torchvision.__version__, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(),
                   'physical_vram_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
                   'before': gpu_status(), 'train_images': 4698, 'validation_images': 524}
    dump(run / 'environment.json', environment)
    positive_weight = (arrays.y['train'] == 0).sum() / (arrays.y['train'] == 1).sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    rows, histories = [], []
    epoch_evidence = {}

    def make_batch(idx, angles):
        x = arrays.x['train'][idx].unsqueeze(1).float().div_(255.)
        theta = torch.zeros((len(idx), 2, 3), device=e.DEVICE)
        theta[:, 0, 0] = theta[:, 1, 1] = angles.cos()
        theta[:, 0, 1], theta[:, 1, 0] = -angles.sin(), angles.sin()
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return (x.repeat(1, 3, 1, 1) - arrays.mean) / arrays.std, arrays.y['train'][idx]

    def execute(seed, batch, session, position):
        rid = f'resnet18_batch{batch}_seed{seed}_session{session}'
        print(f'START {rid}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        e.seed_everything(seed)
        model, _ = e.model_factory('resnet18', pretrained=False)
        start = originals / f'resnet18_shared_seed{seed}.pt'
        model.load_state_dict(torch.load(start, map_location=e.DEVICE, weights_only=True), strict=True)
        start_hash = e.state_hash(model)
        names = e.configure_training(model, 'resnet18', True)
        frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                  if not (k.startswith('layer4.') or k.startswith('fc.'))}
        parameters = [p for p in model.parameters() if p.requires_grad]
        # Disposable forward/backward warms each physical batch shape; reload all weights.
        x, target = make_batch(torch.arange(batch, device=e.DEVICE), torch.zeros(batch, device=e.DEVICE))
        with torch.amp.autocast('cuda', dtype=torch.float16):
            warm_loss = criterion(model(x).view(-1), target)
        warm_loss.backward()
        model.zero_grad(set_to_none=True)
        del x, target, warm_loss
        e.predict(model, arrays, 'val')
        model.load_state_dict(torch.load(start, map_location=e.DEVICE, weights_only=True), strict=True)
        assert e.state_hash(model) == start_hash
        optimizer = torch.optim.AdamW(parameters, lr=PROTOCOL['learning_rate'], weight_decay=PROTOCOL['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
        before_gpu = gpu_status()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        phase_start = time.perf_counter()
        skipped = 0
        for epoch in range(2, 5):
            e.configure_training(model, 'resnet18', True)
            order = torch.randperm(4698, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).to(e.DEVICE)
            gen = torch.Generator(device=e.DEVICE).manual_seed(seed * 1000 + epoch)
            angles = (torch.rand(4698, generator=gen, device=e.DEVICE) * 2 - 1) * (5 * np.pi / 180)
            evidence = hashlib.sha256(order.cpu().numpy().tobytes() + angles.cpu().numpy().tobytes()).hexdigest()
            key = f'{seed}_{epoch}'
            if key in epoch_evidence:
                assert epoch_evidence[key] == evidence, 'Paired augmentation/order mismatch'
            epoch_evidence[key] = evidence
            total = torch.zeros((), device=e.DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for begin in range(0, 4698, batch):
                idx = order[begin:begin + batch]
                x, target = make_batch(idx, angles[begin:begin + batch])
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = criterion(model(x).view(-1), target)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite loss')
                scaler.scale(loss).backward()
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < previous_scale)
                total += loss.detach() * len(idx)
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - t0
            probabilities = e.predict(model, arrays, 'val')
            labels = arrays.y['val'].cpu().numpy()
            score = e.metrics(labels, probabilities)
            record = {'run_id': rid, 'epoch': epoch + 1, 'train_loss': float(total.cpu()) / 4698,
                      'validation_auroc': score['auroc'], 'validation_f1': score['f1'],
                      'training_seconds': training_seconds, 'optimizer_steps': math.ceil(4698 / batch)}
            histories.append(record)
            print('EPOCH', json.dumps(record), flush=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - phase_start
        peak = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        after_gpu = gpu_status()
        for name, tensor in frozen.items():
            assert torch.equal(tensor, model.state_dict()[name].detach().cpu()), f'Frozen tensor changed: {name}'
        checkpoint = run / 'checkpoints' / f'{rid}.pt'
        state_hash = e.state_hash(model)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
        # Reload the actual saved checkpoint, then reproduce its validation predictions.
        model.load_state_dict(torch.load(checkpoint, map_location=e.DEVICE, weights_only=True), strict=True)
        replay = e.predict(model, arrays, 'val')
        replay_difference = float(np.max(np.abs(replay - probabilities)))
        assert replay_difference <= 1e-6
        np.savez_compressed(run / 'predictions' / f'{rid}_validation.npz', indices=arrays.indices['val'], labels=labels, probabilities=probabilities)
        detail = {'run_id': rid, 'start_checkpoint': str(start.relative_to(PROJECT)),
                  'start_file_sha256': sha(start), 'start_state_sha256': start_hash,
                  'final_state_sha256': state_hash, 'checkpoint_sha256': sha(checkpoint),
                  'trainable_names': names, 'gpu_before': before_gpu, 'gpu_after': after_gpu,
                  'frozen_tensors_unchanged': True, 'saved_checkpoint_replay_max_abs_difference': replay_difference}
        dump(run / 'history' / f'{rid}.json', detail)
        row = {'run_id': rid, 'seed': seed, 'batch': batch, 'session': session, 'session_position': position,
               'validation_auroc': score['auroc'], 'validation_f1': score['f1'],
               'validation_accuracy': score['accuracy'], 'adaptation_seconds': seconds,
               'peak_allocated_mib': peak, 'peak_reserved_mib': reserved,
               'steps_per_epoch': math.ceil(4698 / batch), 'total_optimizer_steps': 3 * math.ceil(4698 / batch),
               'skipped_amp_updates': skipped, 'replay_max_abs_difference': replay_difference}
        rows.append(row)
        write_csv(run / 'runs.csv', rows)
        write_csv(run / 'epoch_history.csv', histories)
        print('COMPLETE', json.dumps(row), flush=True)
        del model, optimizer, scaler, parameters, frozen
        gc.collect()
        torch.cuda.empty_cache()

    try:
        for session in PROTOCOL['sessions']:
            for index, seed in enumerate(PROTOCOL['seeds']):
                batches = [16, 32] if (index + session) % 2 else [32, 16]
                for position, batch in enumerate(batches, 1):
                    execute(seed, batch, session, position)
    except Exception as exc:
        dump(run / 'failure.json', {'error': repr(exc), 'completed_runs': len(rows), 'gpu': gpu_status()})
        raise
    assert all(sha(PROJECT / name) == digest for name, digest in original_hashes.items()), 'Original checkpoint changed'
    paired = []
    deterministic = True
    for seed in PROTOCOL['seeds']:
        for batch in [16, 32]:
            hashes = [json.loads((run / 'history' / f'resnet18_batch{batch}_seed{seed}_session{s}.json').read_text())['final_state_sha256'] for s in [1, 2]]
            deterministic &= len(set(hashes)) == 1
        for session in [1, 2]:
            base = next(r for r in rows if r['seed'] == seed and r['session'] == session and r['batch'] == 16)
            candidate = next(r for r in rows if r['seed'] == seed and r['session'] == session and r['batch'] == 32)
            paired.append({'seed': seed, 'session': session,
                           'auroc_delta_batch32_minus16': candidate['validation_auroc'] - base['validation_auroc'],
                           'f1_delta_batch32_minus16': candidate['validation_f1'] - base['validation_f1'],
                           'speedup_batch16_seconds_over32': base['adaptation_seconds'] / candidate['adaptation_seconds'],
                           'allocated_memory_change_mib': candidate['peak_allocated_mib'] - base['peak_allocated_mib']})
    assert deterministic, 'Training did not reproduce across timing sessions; inspect before interpreting results'
    write_csv(run / 'paired_comparisons.csv', paired)
    seed_speedups = [float(np.median([p['speedup_batch16_seconds_over32'] for p in paired if p['seed'] == seed])) for seed in PROTOCOL['seeds']]
    unique_pairs = [p for p in paired if p['session'] == 1]
    delta = float(np.mean([p['auroc_delta_batch32_minus16'] for p in unique_pairs]))
    worst_delta = min(p['auroc_delta_batch32_minus16'] for p in unique_pairs)
    median_speedup = float(np.median(seed_speedups))
    checks = {
        'all_twelve_runs_completed_with_finite_losses': len(rows) == 12,
        'no_skipped_amp_updates': all(r['skipped_amp_updates'] == 0 for r in rows),
        'peak_allocated_memory_below_70_percent': max(r['peak_allocated_mib'] for r in rows) < .7 * environment['physical_vram_mib'],
        'mean_validation_auroc_drop_at_most_0_005': delta >= -.005,
        'each_seed_validation_auroc_drop_at_most_0_010': worst_delta >= -.010,
        'median_paired_speedup_at_least_1_10': median_speedup >= 1.10,
        'technical_sessions_reproduce_final_weights': bool(deterministic),
        'checkpoint_validation_replay_passes': all(r['replay_max_abs_difference'] <= 1e-6 for r in rows),
        'original_checkpoints_unchanged': True,
    }
    accepted = all(checks.values())
    summaries = []
    for batch in [16, 32]:
        first = [r for r in rows if r['batch'] == batch and r['session'] == 1]
        all_runs = [r for r in rows if r['batch'] == batch]
        summaries.append({'batch': batch,
                          'mean_validation_auroc': float(np.mean([r['validation_auroc'] for r in first])),
                          'seed_sd_validation_auroc': float(np.std([r['validation_auroc'] for r in first], ddof=1)),
                          'mean_validation_f1': float(np.mean([r['validation_f1'] for r in first])),
                          'median_adaptation_seconds_technical_runs': float(np.median([r['adaptation_seconds'] for r in all_runs])),
                          'max_allocated_mib': max(r['peak_allocated_mib'] for r in all_runs),
                          'max_reserved_mib': max(r['peak_reserved_mib'] for r in all_runs),
                          'optimizer_steps_per_three_epochs': first[0]['total_optimizer_steps']})
    write_csv(run / 'summary.csv', summaries)
    recommendation = ('Batch 32 meets the predefined engineering pilot criteria; it is a candidate for this ResNet recipe.' if accepted else 'Retain batch 16: batch 32 failed one or more predefined engineering pilot criteria.')
    explained = {'protocol': PROTOCOL, 'summary': summaries, 'checks': checks, 'accepted': bool(accepted),
                 'recommendation': recommendation, 'mean_validation_auroc_delta': delta,
                 'worst_seed_validation_auroc_delta': worst_delta, 'median_paired_speedup': median_speedup,
                 'seed_speedups': seed_speedups,
                 'descriptions': {'validation_auroc': 'Ranking performance on the accepted 524-image validation split; mean and sample SD over three seeds. Timing repetitions do not increase seed count.',
                                  'delta': 'Batch-32 AUROC minus batch-16 AUROC. Negative means worse ranking.',
                                  'speedup': 'Batch-16 adaptation seconds divided by batch-32 seconds. First take the median across two technical sessions for each seed, then the median across three seeds.',
                                  'memory': 'Maximum allocated CUDA tensor memory includes resident train/validation arrays; reserved memory also reported. nvidia-smi before/after snapshots include other GPU uses and are not continuous peaks.',
                                  'acceptance': 'Practical engineering criteria fixed in the original proposal; not a statistical equivalence/non-inferiority test.',
                                  'optimization_scope': 'Changing physical batch changes optimizer update count: 882 versus 441 over the same 14094 image presentations. This is an operational recipe comparison, not a pure kernel-only speed experiment.',
                                  'augmentation_deviation': 'V2 generates all angles once per epoch to guarantee paired per-image rotations across batch sizes. V1 drew angles per batch. Therefore rerun batch16 is the appropriate V2 control; do not substitute old V1 test results.',
                                  'scope_limit': 'ResNet-only, short training budget and existing validation split; no new test evaluation, external validation, or conclusion for DenseNet/MobileNet.'}}
    dump(run / 'results_explained.json', explained)
    dump(run / 'augmentation_pairing.json', epoch_evidence)
    environment['after'] = gpu_status()
    environment['completed_utc'] = e.stamp()
    dump(run / 'environment.json', environment)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for seed in [42, 43, 44]:
        selected = [next(r for r in rows if r['seed'] == seed and r['session'] == 1 and r['batch'] == b) for b in [16, 32]]
        axes[0].plot([16, 32], [r['validation_auroc'] for r in selected], marker='o', label=f'Seed {seed}')
    axes[0].set(xlabel='Physical batch size', ylabel='Validation AUROC', title='Paired predictive results', xticks=[16, 32])
    axes[0].legend(fontsize=8)
    for ax, field, title, label in [(axes[1], 'adaptation_seconds', 'Two timing sessions per seed', 'Adaptation time (seconds)'), (axes[2], 'peak_allocated_mib', 'Allocated CUDA tensor memory', 'Peak allocated MiB')]:
        for batch, offset in [(16, -.15), (32, .15)]:
            values = [r[field] for r in rows if r['batch'] == batch]
            ax.scatter(np.arange(6) + offset, values, label=f'Batch {batch}')
        ax.set(xlabel='Seed/session technical run', ylabel=label, title=title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / 'figures' / 'batch_size_comparison.png', dpi=170)
    plt.close(fig)
    lines = ['# Completed batch-size optimization pilot', '', f"Protocol: `{PROTOCOL['id']}`. Completed UTC: {environment['completed_utc']}.", '',
             f'**Decision: {recommendation}**', '',
             '## Baseline versus candidate', '',
             '| Setting | Validation AUROC mean +/- seed SD | Mean validation F1 | Median adaptation seconds (6 technical runs) | Max allocated MiB | Steps / 3 epochs |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in summaries:
        lines.append(f"| Batch {r['batch']} | {r['mean_validation_auroc']:.6f} +/- {r['seed_sd_validation_auroc']:.6f} | {r['mean_validation_f1']:.6f} | {r['median_adaptation_seconds_technical_runs']:.3f} | {r['max_allocated_mib']:.1f} | {r['optimizer_steps_per_three_epochs']} |")
    lines += ['', f'Mean paired validation AUROC change: **{delta:+.6f}** ({delta * 100:+.4f} percentage points). Worst seed change: {worst_delta:+.6f}.',
              f'Median paired timing speedup: **{median_speedup:.3f}x**; corresponding adaptation-time reduction: {(1 - 1 / median_speedup) * 100:.2f}%.',
              '', '## Per-seed paired changes', '', '| Seed | AUROC change | F1 change | Session 1 speedup | Session 2 speedup |', '| --- | --- | --- | --- | --- |']
    for seed in [42, 43, 44]:
        a, b = [next(p for p in paired if p['seed'] == seed and p['session'] == s) for s in [1, 2]]
        lines.append(f"| {seed} | {a['auroc_delta_batch32_minus16']:+.6f} | {a['f1_delta_batch32_minus16']:+.6f} | {a['speedup_batch16_seconds_over32']:.3f}x | {b['speedup_batch16_seconds_over32']:.3f}x |")
    lines += ['', '## Predefined acceptance checks', '']
    lines += [f"- {'PASS' if passed else 'FAIL'}: {name.replace('_', ' ')}." for name, passed in checks.items()]
    lines += ['', '## Method, interpretation and limits', '',
              'Both settings start from the same saved epoch-2 state per seed, then train layer4 and the output head for three epochs with AdamW lr=0.0001, weight decay=0.0001, weighted BCE and AMP. BatchNorm statistics remain frozen. Head warm-up is not repeated or included in timing. Final epoch is used without validation checkpoint selection. Classification threshold is 0.5.', '',
              'Each seed uses the same shuffled image order and per-image rotations in both branches. Full-epoch angle generation differs from v1 per-batch RNG draws; this version reruns its own batch-16 control. Both settings see 14,094 training-image presentations. Batch 32 makes 441 optimizer updates versus 882 for batch 16. Any speed gain includes this reduction in update count; it is not proof of a faster mathematical optimizer or a universally superior batch size.', '',
              'Two sessions reverse execution order within each seed. Model/validation kernels are warmed using disposable gradients before measured adaptation, then weights are restored. The measured interval includes training, order/augmentation setup and per-epoch validation; it excludes loading and checkpoint writing. The paired speedup is the median of per-seed medians across sessions, not the ratio of the table medians. Thermal and background workload variations remain possible.', '',
              f"GPU: {environment['gpu']}, physical VRAM {environment['physical_vram_mib']:.1f} MiB. The allocated-memory criterion is {environment['physical_vram_mib'] * .7:.1f} MiB. Peak reserved values are in summary.csv; nvidia-smi snapshots are in history records. Allocated tensor memory does not measure every driver allocation or total system peak.", '',
              'The three seeds quantify training variability. Repeated sessions are timing repetitions and reproduce identical final weights; they are not six independent predictive experiments. The tolerances are engineering screens, not statistical non-inferiority/equivalence evidence. No test arrays, test predictions or test labels are loaded. The previously inspected v1 test split remains previously inspected.', '',
              'This optimization concerns ResNet only. It changes neither architecture size nor checkpoint size, and establishes no maximum possible GPU optimization. DenseNet and MobileNet retain their existing validated batch-16 recipe. A failed pilot is retained and reported without post-hoc parameter tuning.', '',
              '## Reproduce and inspect', '',
              'From the project root, run the following using the recorded CUDA Python. A fresh directory is required to train again:', '', '```powershell',
              "& 'C:\\Users\\kanaa\\Documents\\Codex\\ml-gpu-env\\Scripts\\python.exe' -B -u 04_implementation/src/batch32_pilot.py --run-dir 05_experiments/batch32_v2_repeat", '```', '',
              'Verify the retained pilot without training:', '', '```powershell',
              "& 'C:\\Users\\kanaa\\Documents\\Codex\\ml-gpu-env\\Scripts\\python.exe' -B 04_implementation/src/batch32_pilot.py --verify", '```', '',
              'Primary evidence: [runs.csv](runs.csv), [summary.csv](summary.csv), [paired comparisons](paired_comparisons.csv), [epoch history](epoch_history.csv), [explained JSON](results_explained.json), [protocol](protocol.json), [environment](environment.json), [input hashes](inputs.json). Source snapshots are in source/. Final weights are in checkpoints/ and validation predictions in predictions/. Keep the whole project for offline dependencies and data.', '',
              '![Measured validation, timing and memory comparison](figures/batch_size_comparison.png)', '']
    (run / 'OPTIMIZATION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    all_files = sorted(p for p in run.rglob('*') if p.is_file() and p.name != 'integrity.json')
    dump(run / 'integrity.json', {str(p.relative_to(run)): sha(p) for p in all_files})
    verify(run)
    print('FINAL', json.dumps(explained), flush=True)


if __name__ == '__main__':
    main()
