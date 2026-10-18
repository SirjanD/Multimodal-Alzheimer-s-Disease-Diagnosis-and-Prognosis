# Computational Savings Benchmark

## Scope

This benchmark measures the final calibrated logistic-regression stage on the exact saved ADNI2 outer folds. It compares 66 regions (132 MRI/PET inputs) with 15 training-fold GNN-selected regions (30 inputs). Each fitted model was checked against the previously saved probabilities.

The timing excludes image acquisition, PET/MRI registration, MUSE/AAL segmentation, regional feature extraction, and the one-time GNN feature-selection stage. Those operations were unchanged, so the result is evidence of final-model computational savings only.

## Hardware and Software Environment

| Component | Recorded specification |
|---|---|
| Computer | HP Pavilion Laptop 15-eg0xxx |
| Processor | 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz; 4 physical cores, 8 logical processors; reported clock 2803 MHz |
| Memory | 16 GiB installed as 2 x 8 GiB DDR4 3200 MT/s SODIMM modules |
| Available memory at start | 7.88 GiB (50.1% system memory in use) |
| Storage | NVMe SK hynix BC711 HFM512GD3JX013N; 512.11 GB nominal NVMe capacity |
| System-drive free space at start | 19.69 GiB |
| Graphics | Intel(R) Iris(R) Xe Graphics integrated graphics; not used by the timed classifier |
| Operating system | Microsoft Windows 11 Home, build 26200, 64-bit |
| Starting processor state | 2803 MHz reported frequency; 36.7% system CPU load |
| Power state | External power connected: False; battery: 53.0%; active plan: Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced) |
| Python environment | Python 3.11.4; scikit-learn 1.8.0; NumPy 2.4.3; pandas 2.3.3 |
| Accelerator | CPU execution; PyTorch CPU build and no CUDA device; GPU acceleration was not used |

## Timing Method

- Models: class-balanced L2 logistic regression using the liblinear solver.
- Training procedure: four candidate regularisation values were assessed by four-fold inner validation, followed by four-fold training-only sigmoid probability calibration.
- Evaluation workload: five repetitions of five outer folds, producing 25 paired raw-versus-selected timings per task.
- Cohorts: CN versus AD used 159 participants (137 CN and 22 AD); MCI versus AD used 49 participants (27 MCI and 22 AD).
- Fairness controls: both feature sets received an untimed warm-up, their execution order alternated across folds, and GridSearchCV used one worker (n_jobs=1).
- Inference timing: each fitted outer-fold model predicted its held-out batch 500 times; the median batch time was converted to microseconds per participant.
- Timer: Python perf_counter_ns, a monotonic high-resolution wall-clock timer.
- Validation: every recomputed probability was compared with the previously saved fold prediction.
- Uncontrolled conditions: the Windows desktop was not an isolated laboratory compute node; background activity, processor boosting and thermal state may vary.

## Measured Results

| Task | Raw training total | Selected training total | Training reduction | Raw inference | Selected inference | Inference reduction |
|---|---:|---:|---:|---:|---:|---:|
| CN_vs_AD | 8.941 s | 7.214 s | 19.31% (fold-resampling interval 16.56% to 22.17%) | 260.05 us/person | 248.34 us/person | 4.81% (fold-resampling interval 2.39% to 7.26%) |
| MCI_vs_AD | 7.167 s | 6.500 s | 9.30% (fold-resampling interval 5.61% to 13.88%) | 799.43 us/person | 776.82 us/person | 2.86% (fold-resampling interval 1.72% to 3.94%) |

Across both tasks (50 fitted outer-fold models per feature representation), fitting and calibration decreased from 16.108 seconds to 13.715 seconds. This saved 2.393 seconds, corresponding to a pooled final-model reduction of 14.86% on this computer.

The median serialised calibrated-model size decreased from 24.19 KiB to 8.25 KiB, a 65.89% reduction. The numeric input representation itself decreased by 77.27%, from 132 to 30 measurements.

Inference was repeated 500 times for each fitted fold model. Timing intervals describe variation across the 25 saved folds and operating-system conditions; they are not clinical-performance confidence intervals.
Inference time per participant is higher in the smaller MCI-versus-AD batches because fixed pipeline overhead is divided among fewer participants. Comparisons should therefore be made within each task, not between tasks.

## Performance Retained

| Task | Raw 66-region AUC | Selected 15-region AUC | Paired interpretation |
|---|---:|---:|---|
| CN_vs_AD | 97.78% | 98.04% | Performance was retained; superiority was not statistically established |
| MCI_vs_AD | 89.73% | 90.07% | Performance was retained; superiority was not statistically established |

## Interpretation

The 77.27% reduction in regional inputs does not imply a 77.27% reduction in runtime because model-search, calibration, software, and operating-system overhead do not scale linearly with feature count. The measured percentages above should be used instead.

The defensible thesis claim is that the fixed 15-region panel produced potential computational savings in final model fitting and probability calculation while retaining diagnostic discrimination. It is not evidence of reduced scan time, PET tracer cost, segmentation time, total research-development time, or healthcare expenditure.

Absolute time savings were small for these cohorts because both logistic models were already fast. Percentages are specific to this hardware and software environment and should not be generalised as universal runtime reductions.

Maximum absolute difference between recomputed and saved outer-fold probabilities: 5.773e-15.

## Thesis-Ready Statement

On an HP Pavilion Laptop 15-eg0xxx equipped with an Intel Core i7-1165G7 CPU, 16 GiB DDR4-3200 memory and a 512.11 GB NVMe SSD, reducing the multimodal representation from 66 regions (132 MRI/PET measurements) to 15 regions (30 measurements) reduced pooled nested fitting and calibration time from 16.108 seconds to 13.715 seconds. This represents a 14.86% reduction in the final classifier stage on the profiled system while preserving CN-versus-AD AUC (97.78% versus 98.04%) and MCI-versus-AD AUC (89.73% versus 90.07%). The finding indicates potential computational savings, but it does not include image acquisition, registration, segmentation, regional extraction or the one-time GNN feature-selection cost.
