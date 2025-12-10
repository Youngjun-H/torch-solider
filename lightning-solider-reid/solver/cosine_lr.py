"""Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise.

Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import math

import torch

from .scheduler import Scheduler

_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        t_mul: float = 1.0,
        lr_min: float = 0.0,
        decay_rate: float = 1.0,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        cycle_limit=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1."
            )
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(
                    math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul)
                )
                t_i = self.t_mul**i * self.t_initial
                t_curr = t - (1 - self.t_mul**i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate**i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min
                    + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(
                math.floor(
                    -self.t_initial * (self.t_mul**cycles - 1) / (1 - self.t_mul)
                )
            )


class CosineLRSchedulerWrapper(torch.optim.lr_scheduler._LRScheduler):
    """
    PyTorch 표준 API를 따르는 CosineLRScheduler 래퍼
    Lightning과 완전 호환되도록 _LRScheduler를 상속받음
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        t_mul: float = 1.0,
        lr_min: float = 0.0,
        decay_rate: float = 1.0,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        cycle_limit=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        last_epoch=-1,
    ):
        # PyTorch 표준 초기화
        super().__init__(optimizer, last_epoch)

        # 파라미터 저장
        self._t_initial = t_initial
        self._t_mul = t_mul
        self._lr_min = lr_min
        self._decay_rate = decay_rate
        self._warmup_t = warmup_t
        self._warmup_lr_init = warmup_lr_init
        self._warmup_prefix = warmup_prefix
        self._cycle_limit = cycle_limit
        self._t_in_epochs = t_in_epochs
        self._noise_range_t = noise_range_t
        self._noise_pct = noise_pct
        self._noise_std = noise_std
        self._noise_seed = noise_seed

        # base_lrs 저장 (표준 API 요구사항)
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # 커스텀 스케줄러 생성 (initialize=False로 중복 초기화 방지)
        self._custom_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            t_mul=t_mul,
            lr_min=lr_min,
            decay_rate=decay_rate,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            warmup_prefix=warmup_prefix,
            cycle_limit=cycle_limit,
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=False,  # 이미 base_lrs로 초기화됨
        )

    def get_lr(self):
        """PyTorch 표준 API: 현재 epoch에 대한 learning rate 반환"""
        # _custom_scheduler가 아직 생성되지 않은 경우 (초기화 중)
        if not hasattr(self, "_custom_scheduler"):
            return self.base_lrs

        if self._t_in_epochs:
            # epoch 기반: last_epoch은 이미 업데이트됨
            current_epoch = self.last_epoch
            lrs = self._custom_scheduler.get_epoch_values(current_epoch)
            if lrs is None:
                return self.base_lrs
            return lrs
        else:
            # step 기반은 사용하지 않음
            return self.base_lrs

    def step(self, epoch=None):
        """PyTorch 표준 API: step 호출"""
        # _custom_scheduler가 아직 생성되지 않은 경우 (초기화 중)
        if not hasattr(self, "_custom_scheduler"):
            # 초기화 단계에서는 base_lrs를 그대로 유지
            return

        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            epoch = int(epoch)

        self.last_epoch = epoch

        # 커스텀 스케줄러의 step 호출 (이미 optimizer의 lr을 업데이트함)
        self._custom_scheduler.step(epoch=epoch)

        # PyTorch 표준 방식으로도 업데이트 (일관성 유지)
        # _custom_scheduler.step()이 이미 업데이트했으므로
        # get_lr()로 확인만 함
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr
