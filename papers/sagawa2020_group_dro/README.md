# Distributionally robust neural networks

> Status. `in progress`

## Reference

Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto and Percy Liang.
Distributionally robust neural networks. International Conference on
Learning Representations (ICLR 2020). OpenReview `ryxGuJrFvS`,
arXiv:1911.08731, where the title continues "for group shifts: on the
importance of regularization for worst-case generalization".

Citation checked against the OpenReview record in plan step 2b. Section,
table and appendix numbers below follow arXiv v2. The authors' code is at
https://github.com/kohpangwei/group_DRO (commit cbbc1c5).

## Claim

The training distribution is a mixture of $m$ groups $P_g$, and group DRO
minimizes the worst empirical group risk

$$
\hat\theta_{\mathrm{DRO}} = \arg\min_{\theta} \max_{g} \;
    \mathbb{E}_{(x, y) \sim \hat P_g}\big[\ell(\theta; (x, y))\big]
$$

(Eq. 4). On Waterbirds a ResNet-50 classifies landbirds and waterbirds
whose background is land or water, with the background matching the label
for 95% of the training images, so the four groups are the pairs (label,
background). Section 3 compares ERM and group DRO under three regimes. With
the default $\ell_2$ penalty and 300 epochs, both models fit every training
group and still fail on the worst test group. With a strong penalty
$\lambda = 1$ or with early stopping after one epoch, group DRO keeps the
worst group accurate while ERM does not, at a similar average accuracy.

| Claim id | Statement |
| --- | --- |
| c1 | With standard regularization, ERM and group DRO both reach a worst-group training accuracy of 100% (Table 1) |
| c2 | With standard regularization, ERM reaches 97.3% average but 60.0% worst-group test accuracy (Table 1) |
| c3 | With $\lambda = 1$, group DRO raises the worst-group test accuracy from 21.3% for ERM to 84.6% (Table 1) |
| c4 | With early stopping after one epoch, group DRO raises it from 6.7% to 86.0% (Table 1) |
| c5 | In both regularized regimes the average test accuracy of group DRO stays within a few points of ERM (Table 1, Section 3.2) |

The location, metric, paper value and tolerance of each claim are fixed in
`claims.py`, committed before the full run. Table 1 reports one model per
cell, and the worst-group accuracy of a single run varies between seeds,
so c2, c3 and c4 have a tolerance of 15 to 20 points. The tolerance of c5
is the 3 points of the drop that Section 3.2 calls small.

## Setup

| Item | Value |
| --- | --- |
| Data | Waterbirds as released by the authors, read from the Hugging Face dataset `grodino/waterbirds` at revision `e9856c7` (three Parquet files, SHA-256 digests in `data.py`). Its upload script copies every JPEG file of the authors' tarball unchanged with its label, background and split. The training, validation and test sets hold 4,795, 1,199 and 5,794 images, with 56 waterbirds on land in training, as in Section 2.1 |
| Groups | Label times background, the group `2 * y + place` of the authors' `CUBDataset` |
| Model | torchvision ResNet-50 with the `IMAGENET1K_V1` weights (`resnet50-0676ba61.pth`, whose hash torchvision checks on download) and a new linear head with two outputs |
| Input | Resize to 256 by 256, center crop 224, ImageNet normalization, no augmentation (Appendix C.2) |
| Optimizer | SGD with momentum 0.9, batch size 128 and a constant learning rate, the weight decay applied to every parameter |
| Regimes | Standard, learning rate $10^{-3}$, $\lambda = 10^{-4}$, 300 epochs. Strong $\ell_2$ penalty, $10^{-5}$, $\lambda = 1$, 300 epochs. Early stopping, $10^{-3}$, $\lambda = 10^{-4}$, 1 epoch (Appendix C.2) |
| ERM | A random permutation of the training set per epoch |
| Group DRO | `optora.dro.group_losses` and `optora.OnlineGroupDRO` with uniform initial weights and step size 0.01, the authors' `robust_step_size`. An epoch draws 4,795 examples with replacement and every group equally likely, as their `--reweight_groups` sampler does (Algorithm 1) |
| Evaluation | Group accuracies on all three splits in evaluation mode, every 10 epochs and after the last one. The average accuracy weights the group accuracies by the training proportions (Appendix C.1) |
| Seeds | Seed 0 for each 300-epoch run, as Table 1 reports one model per cell. Seeds 0 to 4 for the early-stopping runs |
| Precision | bfloat16 autocast in channels-last format, with the losses and the group weights in float32 |
| Hardware | RTX 3050 Laptop 6 GB |

Choices recorded before any full run.

- Only Waterbirds is reproduced. CelebA has 162,770 training images and
  MultiNLI needs BERT, which take days on this GPU. Table 2 (group
  adjustments) and Table 3 (the grid search with importance weighting) are
  not reproduced, and no claim needs them.
- The authors' tarball on the Stanford server did not answer, and the
  WILDS copy on CodaLab returned no content, so the data comes from the
  Hugging Face copy described above. Its group counts equal the paper's in
  all three splits.
- A batch of 128 does not fit in 6 GB, so every step accumulates the
  gradients of two micro-batches of 64. Batch normalization therefore uses
  the statistics of 64 images. For group DRO the ascent step needs the
  group losses of the whole batch first, so each step evaluates the batch
  once without gradients and without updating the running statistics,
  takes the ascent step, and then backpropagates the weighted loss of each
  micro-batch. Our tests check that this gives the parameters and group
  weights of the authors' `LossComputer` on a full batch.
- The authors trained in float32. bfloat16 autocast runs about 2.2 times
  faster here, 177 against 81 images per second. A training epoch then
  takes about 25 s for ERM and 35 s for group DRO, whose extra pass costs
  the difference, and the whole run about 11 hours.
- The training accuracy of c1 is measured on the final model in evaluation
  mode, while the authors' logs average it over the last epoch in
  training mode.

## Run

```bash
python -m papers.sagawa2020_group_dro.run --config papers/sagawa2020_group_dro/configs/smoke.toml
python -m papers.sagawa2020_group_dro.run --config papers/sagawa2020_group_dro/configs/full.toml
```

The full run saves every finished training run under
`runs/sagawa2020_group_dro/full/`. If it stops, the same command with
`--resume` continues at the first missing run.

## Result

See the results table in the root `README.md`, generated from
`results/sagawa2020_group_dro.<claim_id>.json`.

## Findings

Not run yet.
