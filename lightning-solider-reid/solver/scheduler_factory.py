"""Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""

from .cosine_lr import CosineLRSchedulerWrapper


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = 0.002 * cfg.SOLVER.BASE_LR
    # Warmup 초기값을 더 합리적으로 설정: base_lr의 1%가 아닌 10%로 시작
    # 이렇게 하면 warmup 기간 동안 learning rate가 더 빠르게 증가
    warmup_lr_init = 0.1 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    noise_range = None

    # PyTorch 표준 API를 따르는 래퍼 사용 (Lightning 호환)
    lr_scheduler = CosineLRSchedulerWrapper(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.0,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
    )

    return lr_scheduler
