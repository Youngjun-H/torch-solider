"""
Training Meters for tracking metrics
"""

from typing import Dict, Optional
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricMeter:
    """Collection of AverageMeters for multiple metrics"""

    def __init__(self, metrics: Optional[Dict[str, str]] = None):
        """
        Args:
            metrics: Dict of metric_name -> format_string
        """
        self.meters = {}
        if metrics:
            for name, fmt in metrics.items():
                self.meters[name] = AverageMeter(name, fmt)

    def update(self, values: Dict[str, float], n: int = 1):
        """Update multiple metrics at once"""
        for name, val in values.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(name, ':.4f')
            self.meters[name].update(val, n)

    def reset(self):
        """Reset all meters"""
        for meter in self.meters.values():
            meter.reset()

    def get_avg(self, name: str) -> float:
        """Get average value for a metric"""
        if name in self.meters:
            return self.meters[name].avg
        return 0.0

    def get_val(self, name: str) -> float:
        """Get current value for a metric"""
        if name in self.meters:
            return self.meters[name].val
        return 0.0

    def summary(self) -> str:
        """Get summary string of all metrics"""
        entries = [str(meter) for meter in self.meters.values()]
        return ' | '.join(entries)


class ProgressMeter:
    """Progress meter for epoch training"""

    def __init__(self, num_batches: int, meters: list, prefix: str = ''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> str:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ' '.join(entries)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
