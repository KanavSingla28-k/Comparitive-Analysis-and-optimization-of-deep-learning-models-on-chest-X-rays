"""Run held-out evaluation and resource probes sequentially on one GPU."""
from experiment import evaluate, benchmark
from hardware_probe import probe

if __name__ == '__main__':
    evaluate()
    benchmark()
    probe()
    print('ALL EXPERIMENTS COMPLETE', flush=True)
