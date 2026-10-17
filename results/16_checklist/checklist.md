# Invariance-aware encoding checklist

Which renderings of multivariate machine signals cannot represent which faults.

`blind (proved)` means the rendered images are byte-identical with and without
the fault: no detector, backbone or amount of data can recover it. `blind
(measured)` means every detector tested stayed at chance while the
footprint-matched positive control passed. `untestable` means the encoding is
insensitive to *any* fault of that spatial footprint, so nothing specific can be
concluded about this fault -- which is itself a reason not to use that encoding
for faults of that shape.

## E10 — signal heatmap, per-window min-max (28 ch)

**Do not use for:** per-channel constant offset ((proved)), per-channel gain ((proved)), uniform scaling (raw) ((proved)), uniform scaling (about the mean) ((proved)).

| fault   | fault_description                |   channels_affected | status         |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------------|-------------:|
| S1      | per-channel constant offset      |                   1 | blind (proved) |     0.5      |
| S2      | per-channel gain                 |                   1 | blind (proved) |     0.5      |
| S3      | uniform scaling (raw)            |                 129 | blind (proved) |     0.5      |
| S3b     | uniform scaling (about the mean) |                 129 | blind (proved) |     0.500042 |
| S4      | single-channel time shift        |                   1 | partial        |     0.618435 |
| S4b     | global time shift                |                 129 | visible        |     1        |
| S5      | cycle-duration drift             |                 129 | partial        |     0.593601 |
| S6      | local per-channel offset         |                   1 | partial        |     0.588241 |
| S7      | transient spike, 1 channel       |                   1 | partial        |     0.573006 |
| S7g     | transient spike, all channels    |                 129 | visible        |     1        |
| S7m     | transient spike, 8 channels      |                   8 | visible        |     0.878698 |
| S8      | freeze / hold, 1 channel         |                   1 | partial        |     0.542452 |
| S8g     | freeze / hold, all channels      |                 129 | visible        |     0.993116 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial        |     0.737022 |

## E11 — signal heatmap, train z-score (28 ch)

**No fault in the suite is provably invisible to this encoding.**

| fault   | fault_description                |   channels_affected | status   |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------|-------------:|
| S1      | per-channel constant offset      |                   1 | partial  |     0.610803 |
| S2      | per-channel gain                 |                   1 | partial  |     0.559654 |
| S3      | uniform scaling (raw)            |                 129 | visible  |     1        |
| S3b     | uniform scaling (about the mean) |                 129 | visible  |     0.99205  |
| S4      | single-channel time shift        |                   1 | partial  |     0.614197 |
| S4b     | global time shift                |                 129 | visible  |     1        |
| S5      | cycle-duration drift             |                 129 | partial  |     0.703629 |
| S6      | local per-channel offset         |                   1 | partial  |     0.61367  |
| S7      | transient spike, 1 channel       |                   1 | partial  |     0.570997 |
| S7g     | transient spike, all channels    |                 129 | visible  |     1        |
| S7m     | transient spike, 8 channels      |                   8 | visible  |     0.811468 |
| S8      | freeze / hold, 1 channel         |                   1 | partial  |     0.539681 |
| S8g     | freeze / hold, all channels      |                 129 | visible  |     0.99651  |
| S8m     | freeze / hold, 8 channels        |                   8 | partial  |     0.683684 |

## E1 — signal heatmap, per-window min-max (129 ch)

**Do not use for:** per-channel constant offset ((proved)), per-channel gain ((proved)), uniform scaling (raw) ((proved)), uniform scaling (about the mean) ((proved)).

| fault   | fault_description                |   channels_affected | status         |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------------|-------------:|
| S1      | per-channel constant offset      |                   1 | blind (proved) |     0.5      |
| S2      | per-channel gain                 |                   1 | blind (proved) |     0.5      |
| S3      | uniform scaling (raw)            |                 129 | blind (proved) |     0.500069 |
| S3b     | uniform scaling (about the mean) |                 129 | blind (proved) |     0.500111 |
| S4      | single-channel time shift        |                   1 | partial        |     0.652036 |
| S4b     | global time shift                |                 129 | visible        |     1        |
| S5      | cycle-duration drift             |                 129 | partial        |     0.534931 |
| S6      | local per-channel offset         |                   1 | partial        |     0.578199 |
| S7      | transient spike, 1 channel       |                   1 | partial        |     0.625512 |
| S7g     | transient spike, all channels    |                 129 | visible        |     1        |
| S7m     | transient spike, 8 channels      |                   8 | visible        |     0.915623 |
| S8      | freeze / hold, 1 channel         |                   1 | partial        |     0.53295  |
| S8g     | freeze / hold, all channels      |                 129 | visible        |     0.977202 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial        |     0.645651 |

## E2 — signal heatmap, train z-score (129 ch)

**No fault in the suite is provably invisible to this encoding.**

| fault   | fault_description                |   channels_affected | status   |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------|-------------:|
| S1      | per-channel constant offset      |                   1 | visible  |     0.999612 |
| S2      | per-channel gain                 |                   1 | partial  |     0.663186 |
| S3      | uniform scaling (raw)            |                 129 | visible  |     1        |
| S3b     | uniform scaling (about the mean) |                 129 | visible  |     0.97403  |
| S4      | single-channel time shift        |                   1 | visible  |     0.758837 |
| S4b     | global time shift                |                 129 | visible  |     1        |
| S5      | cycle-duration drift             |                 129 | partial  |     0.654377 |
| S6      | local per-channel offset         |                   1 | visible  |     0.828352 |
| S7      | transient spike, 1 channel       |                   1 | partial  |     0.667632 |
| S7g     | transient spike, all channels    |                 129 | visible  |     1        |
| S7m     | transient spike, 8 channels      |                   8 | visible  |     0.931427 |
| S8      | freeze / hold, 1 channel         |                   1 | partial  |     0.552105 |
| S8g     | freeze / hold, all channels      |                 129 | visible  |     0.999889 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial  |     0.728393 |

## E3 — GASF on PCA-3, per-window min-max

**Do not use for:** per-channel constant offset ((proved)), uniform scaling (raw) ((proved)), uniform scaling (about the mean) ((proved)).

| fault   | fault_description                |   channels_affected | status         |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------------|-------------:|
| S1      | per-channel constant offset      |                   1 | blind (proved) |     0.5      |
| S2      | per-channel gain                 |                   1 | partial        |     0.53374  |
| S3      | uniform scaling (raw)            |                 129 | blind (proved) |     0.5      |
| S3b     | uniform scaling (about the mean) |                 129 | blind (proved) |     0.5      |
| S4      | single-channel time shift        |                   1 | partial        |     0.572465 |
| S4b     | global time shift                |                 129 | visible        |     1        |
| S5      | cycle-duration drift             |                 129 | partial        |     0.551524 |
| S6      | local per-channel offset         |                   1 | partial        |     0.56446  |
| S7      | transient spike, 1 channel       |                   1 | partial        |     0.576427 |
| S7g     | transient spike, all channels    |                 129 | visible        |     0.921053 |
| S7m     | transient spike, 8 channels      |                   8 | partial        |     0.639003 |
| S8      | freeze / hold, 1 channel         |                   1 | partial        |     0.510443 |
| S8g     | freeze / hold, all channels      |                 129 | visible        |     0.966524 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial        |     0.577424 |

## E4 — GASF of 9 channels, 3x3 tiles

**Do not use for:** per-channel constant offset ((proved)), per-channel gain ((proved)), uniform scaling (raw) ((proved)), uniform scaling (about the mean) ((proved)).

| fault   | fault_description                |   channels_affected | status         |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------------|-------------:|
| S1      | per-channel constant offset      |                   1 | blind (proved) |     0.5      |
| S2      | per-channel gain                 |                   1 | blind (proved) |     0.5      |
| S3      | uniform scaling (raw)            |                 129 | blind (proved) |     0.5      |
| S3b     | uniform scaling (about the mean) |                 129 | blind (proved) |     0.5      |
| S4      | single-channel time shift        |                   1 | partial        |     0.538823 |
| S4b     | global time shift                |                 129 | visible        |     1        |
| S5      | cycle-duration drift             |                 129 | partial        |     0.548352 |
| S6      | local per-channel offset         |                   1 | partial        |     0.534903 |
| S7      | transient spike, 1 channel       |                   1 | partial        |     0.512327 |
| S7g     | transient spike, all channels    |                 129 | visible        |     0.836496 |
| S7m     | transient spike, 8 channels      |                   8 | partial        |     0.617535 |
| S8      | freeze / hold, 1 channel         |                   1 | partial        |     0.516593 |
| S8g     | freeze / hold, all channels      |                 129 | visible        |     0.87892  |
| S8m     | freeze / hold, 8 channels        |                   8 | partial        |     0.578324 |

## E5 — recurrence plot, percentile threshold

**Do not use for:** per-channel constant offset ((proved)), uniform scaling (raw) ((proved)), uniform scaling (about the mean) ((proved)).

| fault   | fault_description                |   channels_affected | status         |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------------|-------------:|
| S1      | per-channel constant offset      |                   1 | blind (proved) |     0.5      |
| S2      | per-channel gain                 |                   1 | partial        |     0.514501 |
| S3      | uniform scaling (raw)            |                 129 | blind (proved) |     0.5      |
| S3b     | uniform scaling (about the mean) |                 129 | blind (proved) |     0.5      |
| S4      | single-channel time shift        |                   1 | partial        |     0.617008 |
| S4b     | global time shift                |                 129 | visible        |     0.99615  |
| S5      | cycle-duration drift             |                 129 | partial        |     0.54349  |
| S6      | local per-channel offset         |                   1 | partial        |     0.54126  |
| S7      | transient spike, 1 channel       |                   1 | partial        |     0.54669  |
| S7g     | transient spike, all channels    |                 129 | partial        |     0.704307 |
| S7m     | transient spike, 8 channels      |                   8 | partial        |     0.655928 |
| S8      | freeze / hold, 1 channel         |                   1 | partial        |     0.512978 |
| S8g     | freeze / hold, all channels      |                 129 | visible        |     0.879889 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial        |     0.547008 |

## E6 — recurrence plot, continuous distance

**Do not use for:** per-channel constant offset ((proved)).

| fault   | fault_description                |   channels_affected | status         |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------------|-------------:|
| S1      | per-channel constant offset      |                   1 | blind (proved) |     0.5      |
| S2      | per-channel gain                 |                   1 | partial        |     0.515512 |
| S3      | uniform scaling (raw)            |                 129 | visible        |     1        |
| S3b     | uniform scaling (about the mean) |                 129 | visible        |     1        |
| S4      | single-channel time shift        |                   1 | partial        |     0.624488 |
| S4b     | global time shift                |                 129 | visible        |     1        |
| S5      | cycle-duration drift             |                 129 | partial        |     0.628269 |
| S6      | local per-channel offset         |                   1 | partial        |     0.546953 |
| S7      | transient spike, 1 channel       |                   1 | partial        |     0.622618 |
| S7g     | transient spike, all channels    |                 129 | visible        |     0.929751 |
| S7m     | transient spike, 8 channels      |                   8 | partial        |     0.678504 |
| S8      | freeze / hold, 1 channel         |                   1 | partial        |     0.51338  |
| S8g     | freeze / hold, all channels      |                 129 | visible        |     0.98982  |
| S8m     | freeze / hold, 8 channels        |                   8 | partial        |     0.56072  |

## E7 — STFT log-magnitude, DC kept

**No fault in the suite is provably invisible to this encoding.**

| fault   | fault_description                |   channels_affected | status   |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------|-------------:|
| S1      | per-channel constant offset      |                   1 | partial  |     0.734958 |
| S2      | per-channel gain                 |                   1 | partial  |     0.553158 |
| S3      | uniform scaling (raw)            |                 129 | visible  |     1        |
| S3b     | uniform scaling (about the mean) |                 129 | partial  |     0.615873 |
| S4      | single-channel time shift        |                   1 | partial  |     0.526925 |
| S4b     | global time shift                |                 129 | visible  |     0.999972 |
| S5      | cycle-duration drift             |                 129 | partial  |     0.553296 |
| S6      | local per-channel offset         |                   1 | partial  |     0.547147 |
| S7      | transient spike, 1 channel       |                   1 | partial  |     0.577382 |
| S7g     | transient spike, all channels    |                 129 | visible  |     0.97669  |
| S7m     | transient spike, 8 channels      |                   8 | partial  |     0.684709 |
| S8      | freeze / hold, 1 channel         |                   1 | partial  |     0.508227 |
| S8g     | freeze / hold, all channels      |                 129 | visible  |     0.980443 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial  |     0.567452 |

## E8 — STFT log-magnitude, DC dropped

**No fault in the suite is provably invisible to this encoding.**

| fault   | fault_description                |   channels_affected | status   |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------|-------------:|
| S1      | per-channel constant offset      |                   1 | partial  |     0.571413 |
| S2      | per-channel gain                 |                   1 | partial  |     0.548781 |
| S3      | uniform scaling (raw)            |                 129 | visible  |     1        |
| S3b     | uniform scaling (about the mean) |                 129 | partial  |     0.706898 |
| S4      | single-channel time shift        |                   1 | partial  |     0.540997 |
| S4b     | global time shift                |                 129 | visible  |     1        |
| S5      | cycle-duration drift             |                 129 | partial  |     0.547424 |
| S6      | local per-channel offset         |                   1 | partial  |     0.511731 |
| S7      | transient spike, 1 channel       |                   1 | partial  |     0.582258 |
| S7g     | transient spike, all channels    |                 129 | visible  |     0.977659 |
| S7m     | transient spike, 8 channels      |                   8 | partial  |     0.6891   |
| S8      | freeze / hold, 1 channel         |                   1 | partial  |     0.506717 |
| S8g     | freeze / hold, all channels      |                 129 | visible  |     0.987798 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial  |     0.561925 |

## E9 — GASF on PCA-3, train-global min-max

**No fault in the suite is provably invisible to this encoding.**

| fault   | fault_description                |   channels_affected | status   |   best_auroc |
|:--------|:---------------------------------|--------------------:|:---------|-------------:|
| S1      | per-channel constant offset      |                   1 | partial  |     0.729418 |
| S2      | per-channel gain                 |                   1 | partial  |     0.569474 |
| S3      | uniform scaling (raw)            |                 129 | visible  |     1        |
| S3b     | uniform scaling (about the mean) |                 129 | visible  |     1        |
| S4      | single-channel time shift        |                   1 | partial  |     0.554765 |
| S4b     | global time shift                |                 129 | visible  |     1        |
| S5      | cycle-duration drift             |                 129 | partial  |     0.563019 |
| S6      | local per-channel offset         |                   1 | partial  |     0.543892 |
| S7      | transient spike, 1 channel       |                   1 | partial  |     0.59572  |
| S7g     | transient spike, all channels    |                 129 | visible  |     0.9559   |
| S7m     | transient spike, 8 channels      |                   8 | partial  |     0.634155 |
| S8      | freeze / hold, 1 channel         |                   1 | partial  |     0.511066 |
| S8g     | freeze / hold, all channels      |                 129 | visible  |     0.977659 |
| S8m     | freeze / hold, 8 channels        |                   8 | partial  |     0.565471 |
