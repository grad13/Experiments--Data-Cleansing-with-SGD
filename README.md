# Data Cleansing with SGD-Influence

Reimplementation and bug-fix study of **SGD-influence** — estimating each training
example's influence on a model trained with SGD, and cleansing the training set by
removing harmful examples.

- Original code: https://github.com/sato9hara/sgd-influence
- Paper: Hara, Nitanda, Maehara, *"Data Cleansing for Models Trained with SGD"* (NeurIPS 2019)

## Repository layout

```
reproduction/   Cleaned reimplementation of the original code
extra-1/        Reimplementation with suspected bugs of the original removed
```

Both variants share the same experiment: train a small CNN on MNIST / CIFAR-10 while
snapshotting parameters at every minibatch step, compute per-example SGD-influence
scores by traversing the snapshots in reverse, then retrain with the top-K
harmful examples removed and compare against random removal and a no-removal baseline.

## reproduction/

Pipeline (`run.sh <seed> <gpu> <mnist|cifar10>`):

1. `code/train.py` — SGD training (20 epochs), saving model parameters before every
   minibatch and per-epoch metrics (`score.dat`, `config.json`).
2. `code/infl.py` — batched per-example influence scores via `exgrads`, traversing the
   saved steps in reverse from the validation-set gradient
   (`infl_sgd_at_epoch20.dat`).
3. `code/eval.py` — retrains after removing top-K examples ranked by `sgd_last` /
   `sgd_all` / `random` for K = 1…10000 (`eval_epoch_19_to_20.dat`).
4. `visualize/sumup.py` (run separately) — averages test error over trials under
   `MIRROR/` and plots error vs. number of removed examples (`VIEWs/*.png`).

Stated differences from the original (none of which should affect results):

1. Random seed is fixed for `numpy` and `pytorch` too, not just `random`.
2. Influence scores are computed in batch rather than per example.
3. Parameter dumping is simplified.
4. Data augmentation is applied once at the start of each epoch.
5. The optimizer is assumed to be SGD with `momentum=0`. (Note: `train.py` still
   writes per-epoch optimizer snapshots — `epochNN_optimizer.dat`.)

## extra-1/

Same experiment, but with four suspected bugs of the original implementation fixed
(details and code references in `extra-1/README.md`):

1. **Influence update scale** — the query update was wrongly divided by the
   mini-batch size (`/ idx.size` removed).
2. **BatchNorm drift during scoring** — scoring-only forward passes mutated BatchNorm
   running statistics; `track_running_stats` is now disabled while scoring.
3. **SGD update scale in counterfactual retraining** — the original multiplied
   mean-reduced gradients by `t / idx.size`, double-scaling the step.
4. **Mini-batch composition when removing examples** — instead of dropping removed
   examples from the batch, the full batch is forwarded and removed examples are
   masked out of the (unreduced) loss, so BatchNorm sees the original batch.

Additionally, random-crop padding is filled with the normalized value of a raw zero
pixel (`-mean/std`) instead of a literal zero in normalized space.

Pipeline (`run.py`, distributes jobs over GPUs 0–7 via `query_queue_and_parallel`):

1. `code/1-train.py` — both datasets × seeds 0–7 × 20 epochs
   (`params/NN.pt`, `trails.pkl`, `config.json`).
2. `code/2-score.py` — influence scores per episode (`scores/20.pkl`).
3. `code/3-evaluate.py` — counterfactual retraining from `start_epoch` ∈ {0, 19}
   (`evaluations/00_to_20.pkl`, `19_to_20.pkl`).
4. `visualize/sumup.py` — plots under `VIEW/data=<dataset>-start_epoch=<NN>.png`,
   where `start_epoch=00` replays all 20 epochs and `start_epoch=19` replays only the
   last epoch.

## Results

- `reproduction/VIEWs/{mnist,cifar10}.png`
- `extra-1/VIEW/data={mnist,cifar10}-start_epoch={00,19}.png`

## Requirements

- Python with `torch`, `torchvision`, `numpy`, `joblib`, `matplotlib`, `exgrads`,
  and (for `extra-1/run.py`) `query_queue_and_parallel`.
- **CUDA is required** — all data and models are moved with `.cuda(gpu)`; there is no
  CPU fallback. cuDNN is set to deterministic mode throughout.
- Datasets are downloaded and cached automatically under `data/`.
