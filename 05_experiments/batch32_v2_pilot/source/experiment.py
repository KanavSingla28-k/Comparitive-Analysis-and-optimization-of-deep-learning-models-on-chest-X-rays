"""Training, models, preprocessing and inference for the frozen college benchmark. See settings.py."""
from __future__ import annotations

from runtime import ROOT, ART, PROJECT_ROOT, DATA_FILE
from settings import CONFIG

import sys
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import models
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)



DEVICE = torch.device('cuda')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA must be available; activate the recorded GPU environment.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for sub in ['checkpoints', 'predictions', 'history', 'figures', 'benchmarks']:
        (ART / sub).mkdir(parents=True, exist_ok=True)


def model_factory(name, pretrained=True):
    if name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == 'densenet121':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif name == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:
        raise ValueError(name)
    return model.to(DEVICE), str(weights)


def configure_training(model, name, partial):
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.fc if name == 'resnet18' else model.classifier
    for p in head.parameters():
        p.requires_grad_(True)
    if partial:
        if name == 'resnet18':
            modules = [model.layer4]
        elif name == 'densenet121':
            modules = [model.features.denseblock4, model.features.norm5]
        else:
            modules = [model.features[-2], model.features[-1]]
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)
    # Frozen running statistics are intentional, including in trainable stages.
    model.eval()
    head.train()
    return [n for n, p in model.named_parameters() if p.requires_grad]


def state_hash(model):
    h = hashlib.sha256()
    for n, t in model.state_dict().items():
        h.update(n.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def raw_data():
    path = DATA_FILE
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def audit_data():
    path = DATA_FILE
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != CONFIG['dataset_md5']:
        raise RuntimeError('Dataset checksum mismatch; do not train.')
    data = raw_data()
    hashes = {}
    summary = {}
    for split in ['train', 'val', 'test']:
        images, labels = data[f'{split}_images'], data[f'{split}_labels'].reshape(-1)
        assert images.shape[1:] == (224, 224)
        assert set(np.unique(labels)) == {0, 1}
        hashes[split] = np.array([hashlib.sha256(a.tobytes()).hexdigest() for a in images])
        summary[split] = {'images': len(labels), 'shape': list(images.shape),
                          'normal_0': int((labels == 0).sum()),
                          'pneumonia_1': int((labels == 1).sum()),
                          'within_split_duplicate_images': len(labels) - len(set(hashes[split]))}
    # Audit only: test images/labels do not enter training, threshold or model choices.
    keep = {}
    test_hashes = set(hashes['test'])
    val_mask = np.array([h not in test_hashes for h in hashes['val']])
    val_hashes = set(hashes['val'][val_mask])
    train_mask = np.array([h not in test_hashes and h not in val_hashes for h in hashes['train']])
    keep['train'] = np.flatnonzero(train_mask)
    keep['val'] = np.flatnonzero(val_mask)
    keep['test'] = np.arange(len(hashes['test']))
    for split in keep:
        np.save(ART / f'{split}_indices.npy', keep[split])
        summary[split]['retained'] = len(keep[split])
    overlaps = {f'{a}_{b}': len(set(hashes[a]) & set(hashes[b]))
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    result = {'created_utc': stamp(), 'url': CONFIG['dataset_url'], 'md5': h.hexdigest(),
              'file_bytes': path.stat().st_size, 'array_keys': list(data),
              'class_mapping_source': 'MedMNIST official metadata: 0 normal, 1 pneumonia',
              'splits': summary, 'cross_split_unique_exact_overlaps': overlaps,
              'patient_ids_available': False,
              'limitation': 'No patient IDs in NPZ. Exact image hashes do not establish patient independence.',
              'cleanup': 'Remove development images identical to test; remove train images identical to retained validation.'}
    write_json(ART / 'dataset_audit.json', result)
    print('DATA AUDIT', json.dumps(result), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
    labels = data['train_labels'].reshape(-1)
    for row, label in enumerate([0, 1]):
        for ax, idx in zip(axes[row], np.flatnonzero(labels == label)[:6]):
            ax.imshow(data['train_images'][idx], cmap='gray')
            ax.set_title('Normal' if label == 0 else 'Pneumonia')
            ax.axis('off')
    fig.suptitle('Training examples only — PneumoniaMNIST+ 224')
    fig.tight_layout()
    fig.savefig(ART / 'figures' / 'training_examples.png', dpi=150)
    plt.close(fig)


class ImageArrays:
    def __init__(self, splits):
        data = raw_data()
        self.x, self.y, self.indices = {}, {}, {}
        for split in splits:
            idx = np.load(ART / f'{split}_indices.npy')
            self.indices[split] = idx
            self.x[split] = torch.from_numpy(data[f'{split}_images'][idx]).to(DEVICE)
            self.y[split] = torch.from_numpy(data[f'{split}_labels'][idx].reshape(-1).astype(np.float32)).to(DEVICE)
        self.mean = torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]

    def batch(self, split, indices, generator=None):
        x = self.x[split][indices].unsqueeze(1).float().div_(255.)
        if generator is not None:
            angle = (torch.rand(len(indices), generator=generator, device=DEVICE) * 2 - 1)
            angle *= CONFIG['rotation_degrees'] * np.pi / 180
            theta = torch.zeros((len(indices), 2, 3), device=DEVICE)
            theta[:, 0, 0] = theta[:, 1, 1] = angle.cos()
            theta[:, 0, 1] = -angle.sin()
            theta[:, 1, 0] = angle.sin()
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std, self.y[split][indices]


@torch.inference_mode()
def predict(model, arrays, split, amp=True, batch_size=64):
    model.eval()
    out = []
    for start in range(0, len(arrays.y[split]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(arrays.y[split])), device=DEVICE)
        x, _ = arrays.batch(split, indices)
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = model(x).view(-1)
        out.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    pred = p >= CONFIG['fixed_threshold']
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auroc': float(roc_auc_score(y, p)),
            'ap': float(average_precision_score(y, p)),
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def train_phase(model, name, arrays, seed, partial, start_epoch, epochs, shared_history=None):
    names = configure_training(model, name, partial)
    lr = CONFIG['head_lr'] if start_epoch == 0 else CONFIG['adaptation_lr']
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    y = arrays.y['train']
    positive_weight = (y == 0).sum() / (y == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    n = len(y)
    history = list(shared_history or [])
    torch.cuda.reset_peak_memory_stats()
    phase_start = time.perf_counter()
    frozen_before = model.conv1.weight.detach().clone() if name == 'resnet18' else None
    for epoch in range(start_epoch, start_epoch + epochs):
        configure_training(model, name, partial)
        cpu_generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        aug_generator = torch.Generator(device=DEVICE).manual_seed(seed * 1000 + epoch)
        order = torch.randperm(n, generator=cpu_generator).to(DEVICE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total_loss = torch.zeros((), device=DEVICE)
        for begin in range(0, n, CONFIG['batch_size']):
            idx = order[begin:begin + CONFIG['batch_size']]
            x, target = arrays.batch('train', idx, aug_generator)
            # Replaying dropout randomness makes the paired branch protocol explicit.
            torch.cuda.manual_seed(seed * 100000 + epoch * 1000 + begin)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x).view(-1)
                loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * len(idx)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - t0
        vp = predict(model, arrays, 'val')
        vy = arrays.y['val'].cpu().numpy()
        val_loss = float(F.binary_cross_entropy(torch.from_numpy(vp).clamp(1e-7, 1-1e-7), torch.from_numpy(vy)))
        row = {'epoch': epoch + 1, 'phase': 'partial' if partial else 'head',
               'train_loss': float(total_loss.cpu()) / n,
               'val_unweighted_bce': val_loss, 'val_auroc': float(roc_auc_score(vy, vp)),
               'val_ap': float(average_precision_score(vy, vp)),
               'training_seconds': train_seconds, 'training_images_per_second': n / train_seconds,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        history.append(row)
        print('EPOCH', name, seed, json.dumps(row), flush=True)
    torch.cuda.synchronize()
    if frozen_before is not None:
        assert torch.equal(frozen_before, model.conv1.weight), 'Frozen backbone changed'
    details = {'seconds_including_validation': time.perf_counter() - phase_start,
               'peak_allocated_mib_including_data': torch.cuda.max_memory_allocated() / 2**20,
               'trainable_names': names, 'positive_weight': float(positive_weight.cpu())}
    return history, details


def save_run(model, run_id, name, seed, weights, history, phase_details, initial_hash=None):
    path = ART / 'checkpoints' / f'{run_id}.pt'
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)
    write_json(ART / 'history' / f'{run_id}.json', history)
    result = {'id': run_id, 'architecture': name, 'seed': seed, 'weights': weights,
              'completed_utc': stamp(), 'epochs': len(history),
              'parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_bytes_fp32': path.stat().st_size,
              'phase_details': phase_details, 'shared_epoch2_hash': initial_hash,
              'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'total_seconds_including_validation': sum(d['seconds_including_validation'] for d in phase_details),
              'final_validation_auroc': history[-1]['val_auroc']}
    write_json(ART / 'history' / f'{run_id}_manifest.json', result)
    print('RUN COMPLETE', json.dumps({k: v for k, v in result.items() if k != 'phase_details'}), flush=True)


def prepare():
    setup()
    env = {'created_utc': stamp(), 'python': sys.version, 'executable': sys.executable,
           'torch': torch.__version__, 'torchvision': torchvision.__version__,
           'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'gpu_vram_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
           'platform': platform.platform(), 'cudnn': torch.backends.cudnn.version(),
           'cpu_threads': torch.get_num_threads(), 'tf32': False,
           'package_location': str(PROJECT_ROOT / 'archive' / 'pilot_2026-10-22' / '.deps')}
    write_json(ART / 'environment.json', env)
    import importlib.metadata
    packages = sorted({f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata['Name']})
    (ROOT / 'requirements-lock.txt').write_text('\n'.join(packages) + '\n')
    audit_data()
    arrays = ImageArrays(['train', 'val'])
    smoke = []
    for name in CONFIG['architectures']:
        seed_everything(42)
        model, weights = model_factory(name)
        ids = torch.arange(CONFIG['batch_size'], device=DEVICE)
        x, y = arrays.batch('train', ids)
        for partial in [False, True]:
            trainable = configure_training(model, name, partial)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            model.zero_grad(set_to_none=True)
            scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(x)
                assert out.shape == (CONFIG['batch_size'], 1)
                loss = F.binary_cross_entropy_with_logits(out.view(-1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            frozen_ok = all(torch.equal(before[n], p) for n, p in model.named_parameters() if not p.requires_grad)
            changed = sum(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
            assert frozen_ok and changed > 0, f'Frozen unchanged={frozen_ok}, updated tensors={changed}, scaler={scaler.get_scale()}'
            smoke.append({'architecture': name, 'partial': partial, 'loss': float(loss),
                          'seconds': time.perf_counter() - t0, 'frozen_weights_unchanged': frozen_ok,
                          'trainable_tensors_changed': changed,
                          'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                          'weights': weights, 'trainable_names': trainable})
            print('SMOKE OK', name, partial, flush=True)
            del before, optimizer, out, loss
        del model
        torch.cuda.empty_cache()
    write_json(ART / 'smoke_test.json', smoke)
    write_json(ART / 'protocol_lock.json', {'created_utc': stamp(), 'config': CONFIG,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'test_predictions_generated': False})
    print('PREPARE COMPLETE', json.dumps(env), flush=True)


def train(seeds):
    setup()
    if not (ART / 'protocol_lock.json').exists():
        raise RuntimeError('Run prepare before training')
    arrays = ImageArrays(['train', 'val'])
    for seed in seeds:
        for name in CONFIG['architectures']:
            partial_id = f'{name}_partial_seed{seed}'
            frozen_id = f'{name}_frozen_seed{seed}'
            if (ART / 'history' / f'{partial_id}_manifest.json').exists() and (name != 'resnet18' or (ART / 'history' / f'{frozen_id}_manifest.json').exists()):
                print('SKIP COMPLETE', name, seed, flush=True)
                continue
            seed_everything(seed)
            model, weights = model_factory(name)
            shared_history, head_details = train_phase(model, name, arrays, seed, False, 0, CONFIG['head_epochs'])
            shared_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            shared_hash = state_hash(model)
            if name == 'resnet18':
                torch.save(shared_state, ART / 'checkpoints' / f'resnet18_shared_seed{seed}.pt')
                history, frozen_details = train_phase(model, name, arrays, seed, False, CONFIG['head_epochs'], CONFIG['adaptation_epochs'], shared_history)
                save_run(model, frozen_id, name, seed, weights, history, [head_details, frozen_details], shared_hash)
                model.load_state_dict(shared_state)
                assert state_hash(model) == shared_hash
            history, partial_details = train_phase(model, name, arrays, seed, True, CONFIG['head_epochs'], CONFIG['adaptation_epochs'], shared_history)
            save_run(model, partial_id, name, seed, weights, history, [head_details, partial_details], shared_hash)
            del model, shared_state
            torch.cuda.empty_cache()
    print('TRAINING COMPLETE', flush=True)


def evaluate():
    setup()
    expected = [(name, seed, variant) for seed in CONFIG['seeds'] for name in CONFIG['architectures']
                for variant in (['frozen', 'partial'] if name == 'resnet18' else ['partial'])]
    for name, seed, variant in expected:
        rid = f'{name}_{variant}_seed{seed}'
        if not (ART / 'history' / f'{rid}_manifest.json').exists():
            raise RuntimeError(f'Complete all declared runs before evaluating test: {rid}')
    arrays = ImageArrays(['test'])
    y = arrays.y['test'].cpu().numpy()
    rows = []
    for name, seed, variant in expected:
        rid = f'{name}_{variant}_seed{seed}'
        model, _ = model_factory(name, pretrained=False)
        model.load_state_dict(torch.load(ART / 'checkpoints' / f'{rid}.pt', map_location=DEVICE, weights_only=True))
        p = predict(model, arrays, 'test', amp=True)
        pf = predict(model, arrays, 'test', amp=False)
        np.savez(ART / 'predictions' / f'{rid}.npz', indices=arrays.indices['test'], labels=y, probabilities=p, fp32_probabilities=pf)
        row = {'id': rid, 'architecture': name, 'variant': variant, 'seed': seed,
               **metrics(y, p), 'fp32_auroc': float(roc_auc_score(y, pf)),
               'amp_fp32_max_probability_difference': float(np.max(np.abs(p-pf))),
               'amp_fp32_decision_disagreements': int(np.sum((p >= .5) != (pf >= .5)))}
        rows.append(row)
        print('TEST', json.dumps(row), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(ART / 'test_metrics.json', rows)
    with (ART / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def timed_forward(model, x, amp, warmup=30, repeats=200):
    for _ in range(warmup):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
            model(x)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    # One context per forward reflects isolated prediction calls, without data transfer.
    start_wall = time.perf_counter()
    for start, end in events:
        start.record()
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
            model(x)
        end.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - start_wall
    times = np.array([a.elapsed_time(b) for a, b in events])
    return {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
            'mean_ms': float(times.mean()), 'wall_images_per_second': len(x) * repeats / wall,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'incremental_peak_mib': (torch.cuda.max_memory_allocated() - baseline) / 2**20,
            'timings_ms': times.tolist(), 'repeats': repeats, 'warmup': warmup}


def benchmark():
    setup()
    output = []
    for name in CONFIG['architectures']:
        rid = f'{name}_partial_seed42'
        model, _ = model_factory(name, pretrained=False)
        model.load_state_dict(torch.load(ART / 'checkpoints' / f'{rid}.pt', weights_only=True, map_location=DEVICE))
        model.eval()
        # Three interleaved precision rounds reduce warm-up/order confounding.
        for round_id in range(3):
            modes = [False, True] if round_id % 2 == 0 else [True, False]
            for amp in modes:
                x = torch.randn(1, 3, 224, 224, device=DEVICE)
                result = timed_forward(model, x, amp)
                row = {'architecture': name, 'batch_size': 1, 'amp': amp, 'round': round_id, **result}
                output.append(row)
                print('BENCH', json.dumps({k: v for k, v in row.items() if k != 'timings_ms'}), flush=True)
                del x
        for batch in [16, 32, 64, 128]:
            x = None
            try:
                x = torch.randn(batch, 3, 224, 224, device=DEVICE)
                result = timed_forward(model, x, True, warmup=5, repeats=30)
                row = {'architecture': name, 'batch_size': batch, 'amp': True, 'round': 0, **result}
                output.append(row)
                print('BATCH BENCH', name, batch, result['wall_images_per_second'], result['peak_allocated_mib'], flush=True)
                if result['peak_allocated_mib'] > 0.7 * torch.cuda.get_device_properties(0).total_memory / 2**20:
                    break
            except torch.OutOfMemoryError:
                output.append({'architecture': name, 'batch_size': batch, 'amp': True, 'error': 'CUDA OOM'})
                break
            finally:
                del x
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    write_json(ART / 'benchmarks' / 'inference.json', output)
    print('BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'benchmark'])
    parser.add_argument('--seeds', default=','.join(map(str, CONFIG['seeds'])))
    args = parser.parse_args()
    {'prepare': prepare, 'train': lambda: train([int(s) for s in args.seeds.split(',')]),
     'evaluate': evaluate, 'benchmark': benchmark}[args.command]()
