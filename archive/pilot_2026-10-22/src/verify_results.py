"""Independent artifact consistency checks, using saved probabilities and files."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
import hashlib
import json
import numpy as np
from scipy.stats import rankdata

ART = ROOT / 'artifacts'
config = json.loads((ROOT / 'configs/experiment.json').read_text())
lock = json.loads((ART / 'protocol_lock.json').read_text())
assert lock['source_sha256'] == hashlib.sha256((ROOT / 'src/experiment.py').read_bytes()).hexdigest()
assert config == lock['config']
assert lock['test_predictions_generated'] is False
rows = json.loads((ART / 'test_metrics.json').read_text())
assert len(rows) == 12
hash_checks = []
for row in rows:
    rid = row['id']
    pfile = np.load(ART / 'predictions' / f'{rid}.npz')
    y, p = pfile['labels'], pfile['probabilities']
    assert len(y) == 624 and set(np.unique(y)) == {0, 1}
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    npos = (y == 1).sum()
    auc = (rankdata(p)[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))
    assert abs(auc - row['auroc']) < 1e-12
    pred = p >= .5
    tn, fp = int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())
    fn, tp = int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())
    assert (tn, fp, fn, tp) == tuple(row[k] for k in ['tn', 'fp', 'fn', 'tp'])
    assert abs(2*tp / (2*tp+fp+fn) - row['f1']) < 1e-12
    manifest = json.loads((ART / 'history' / f'{rid}_manifest.json').read_text())
    hist = json.loads((ART / 'history' / f'{rid}.json').read_text())
    assert [r['epoch'] for r in hist] == [1, 2, 3, 4, 5]
    assert all(np.isfinite(r['train_loss']) for r in hist)
    sha = hashlib.sha256((ART / 'checkpoints' / f'{rid}.pt').read_bytes()).hexdigest()
    assert sha == manifest['checkpoint_sha256']
    hash_checks.append(rid)
for seed in config['seeds']:
    a = json.loads((ART / 'history' / f'resnet18_frozen_seed{seed}_manifest.json').read_text())
    b = json.loads((ART / 'history' / f'resnet18_partial_seed{seed}_manifest.json').read_text())
    assert a['shared_epoch2_hash'] == b['shared_epoch2_hash']
with np.load(ROOT / 'data/pneumoniamnist_224.npz') as data:
    clean_hashes = {}
    for split in ['train', 'val', 'test']:
        indices = np.load(ART / f'{split}_indices.npy')
        clean_hashes[split] = {hashlib.sha256(x.tobytes()).hexdigest() for x in data[f'{split}_images'][indices]}
    assert not (clean_hashes['train'] & clean_hashes['val'])
    assert not (clean_hashes['train'] & clean_hashes['test'])
    assert not (clean_hashes['val'] & clean_hashes['test'])
out = {'completed_runs_verified': len(rows), 'checkpoint_hashes_verified': hash_checks,
       'auroc_independently_recomputed_from_ranks': True,
       'confusion_counts_and_f1_recomputed': True,
       'paired_initial_checkpoint_hashes_match': True,
       'post_cleanup_exact_cross_split_overlaps': 0,
       'training_source_and_config_match_protocol_lock': True}
(ART / 'verification.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
