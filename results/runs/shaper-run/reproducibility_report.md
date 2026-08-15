# Reproducibility report

- run_id: `shaper-run`
- dataset: `ml100k`
- config hash: `71a3d06011d8dcc791d6f8808d93c958`
- python (verified interpreter): `3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 21.0.0 (clang-2100.0.123.102)]`
- torch: `2.13.0`; CUDA available: `False`
- OS: `Darwin 25.6.0`; CPU count: `12`
- repository commit: `df3c89876feb4a747a6824359734b1703c8369f7`; dirty: `True`
- deterministic flags: `{"cudnn_benchmark": false, "cudnn_deterministic": true, "torch_deterministic_algorithms": true}`
- peak allocated GPU memory: `None` bytes
- peak reserved GPU memory: `None` bytes

## Lock file

Exact resolved dependency versions are recorded in `requirements.lock`
(authoring constraint: Python 3.12, torch >= 2.4, numpy >= 2.4,<2.5,
scipy >= 1.18; the CI sandbox verified the code on Python 3.11 with the
newest compatible SciPy, recorded in environment.json).

## Determinism

`torch.use_deterministic_algorithms(True)` with cuDNN benchmarking disabled
and deterministic cuDNN enabled (see preflight). The canonical grand
coalition is repeated on seed 2001 and the absolute utility difference
is reported as the execution-nondeterminism floor; contextual marginals
whose magnitude does not exceed the floor are labeled indistinguishable
from execution nondeterminism.

## Random schedules

Augmentation draws are keyed by (seed, optimizer_step, global_example_id,
occurrence, view); recommendation negatives by (seed, optimizer_step,
global_user_id, target_position); epoch permutations by (seed, epoch,
dataset_hash); dropout forwards by (seed, optimizer_step, purpose, view,
pass_index). Worker count and coalition enumeration order cannot change
any schedule.

