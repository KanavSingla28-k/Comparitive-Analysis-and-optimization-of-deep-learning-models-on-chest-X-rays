"""Executable study settings. Change the protocol version before changing the recipe.

These constants, rather than JSON configuration, drive training and analysis.
"""
PROTOCOL_ID = 'cxr_compare_v1'
CONFIG = {
    'dataset': 'PneumoniaMNIST+ 224',
    'dataset_url': 'https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1',
    'dataset_md5': 'd6a3c71de1b945ea11211b03746c1fe1',
    'architectures': ['resnet18', 'densenet121', 'mobilenet_v3_small'],
    'seeds': [42, 43, 44],
    'batch_size': 16,
    'head_epochs': 2,
    'adaptation_epochs': 3,
    'head_lr': 0.001,
    'adaptation_lr': 0.0001,
    'weight_decay': 0.0001,
    'rotation_degrees': 5,
    'fixed_threshold': 0.5,
    'precision': 'CUDA autocast float16 with GradScaler',
    'batchnorm': 'running statistics frozen in all phases',
    'checkpoint_selection': 'fixed final epoch, no test selection',
    'optimization': 'paired ResNet frozen vs partial fine-tuning; FP32 vs AMP inference',
    'bootstrap_repetitions': 2000,
}
DATASET_SHA256 = '2a6faaca43872b4bcf4154a79625631ab29683b41e265ab7936d0513351fe591'
BOOTSTRAP_SEED = 20261022
GROUPS = ['resnet18_frozen', 'resnet18_partial', 'densenet121_partial',
          'mobilenet_v3_small_partial']
LABELS = ['ResNet-18 frozen', 'ResNet-18 partial FT', 'DenseNet-121 partial FT',
          'MobileNetV3-Small partial FT']
PAIRED_COMPARISONS = [(1, 0), (1, 2), (3, 2)]