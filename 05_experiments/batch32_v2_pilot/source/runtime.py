"""Bind data, caches and versioned outputs without modifying the pilot archive."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = PROJECT_ROOT / 'archive' / 'pilot_2026-10-22'
# A local installation takes precedence. Bundled pilot libraries are an offline fallback.
if sys.prefix == sys.base_prefix or not (PROJECT_ROOT / '.venv').exists():
    bundled = ARCHIVE / '.deps'
    if bundled.exists():
        sys.path.insert(0, str(bundled))
RUN_ROOT = Path(os.environ.get('IRS_RUN_DIR', str(PROJECT_ROOT / '05_experiments' / 'reproduced_v1'))).resolve()
if not RUN_ROOT.is_relative_to(PROJECT_ROOT / '05_experiments') or RUN_ROOT == PROJECT_ROOT / '05_experiments':
    raise ValueError('Run directory must be a child of this project / 05_experiments')
DATA_FILE = Path(os.environ.get('IRS_DATA_FILE', str(ARCHIVE / 'data' / 'pneumoniamnist_224.npz'))).resolve()
if not DATA_FILE.is_relative_to(PROJECT_ROOT):
    raise ValueError('Keep the dataset inside this project root')
ART = RUN_ROOT / 'artifacts'
ROOT = RUN_ROOT
os.environ['TORCH_HOME'] = str(PROJECT_ROOT / '.cache' / 'torch')
os.environ['MPLCONFIGDIR'] = str(PROJECT_ROOT / '.cache' / 'matplotlib')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')