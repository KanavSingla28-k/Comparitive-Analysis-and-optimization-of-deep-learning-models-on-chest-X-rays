"""Synthetic training capacity probe; never changes saved research checkpoints."""
from experiment import *


def probe():
    setup()
    output = []
    for name in CONFIG['architectures']:
        checkpoint = torch.load(ART / 'checkpoints' / f'{name}_partial_seed42.pt', map_location='cpu', weights_only=True)
        for batch, amp in [(16, False), (16, True), (32, True), (64, True)]:
            model = optimizer = scaler = x = y = loss = logits = None
            try:
                seed_everything(17)
                model, _ = model_factory(name, pretrained=False)
                model.load_state_dict(checkpoint)
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
                model.eval()  # Same fixed BN stats as the comparative protocol.
                x = torch.randn(batch, 3, 224, 224, device=DEVICE)
                y = (torch.arange(batch, device=DEVICE) % 2).float()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=amp)
                torch.cuda.reset_peak_memory_stats()
                times = []
                for step in range(13):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp):
                        logits = model(x).view(-1)
                        loss = F.binary_cross_entropy_with_logits(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    if step >= 3:
                        times.append(time.perf_counter() - t0)
                peak = torch.cuda.max_memory_allocated() / 2**20
                row = {'architecture': name, 'batch_size': batch, 'amp': amp,
                       'scope': 'synthetic full-gradient AdamW step, fixed BN stats, no preprocessing/transfer',
                       'median_step_ms': float(np.median(times) * 1000),
                       'images_per_second': float(batch / np.median(times)),
                       'peak_allocated_mib': peak, 'warmup_steps': 3, 'measured_steps': 10,
                       'times_seconds': times}
                output.append(row)
                print('TRAIN CAPACITY', json.dumps({k: v for k, v in row.items() if k != 'times_seconds'}), flush=True)
                if peak > .7 * torch.cuda.get_device_properties(0).total_memory / 2**20:
                    break
            except torch.OutOfMemoryError:
                output.append({'architecture': name, 'batch_size': batch, 'amp': amp, 'error': 'CUDA OOM'})
                break
            finally:
                del model, optimizer, scaler, x, y, loss, logits
                torch.cuda.empty_cache()
        del checkpoint
    write_json(ART / 'benchmarks' / 'training_capacity.json', output)
    print('TRAINING CAPACITY PROBE COMPLETE', flush=True)


if __name__ == '__main__':
    probe()
