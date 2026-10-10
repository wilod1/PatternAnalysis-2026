# ConvNeXt on ADNI: AD vs CN classification
## Problem and algorithm
## How it works
## Feasibility review
## Dependencies and reproducibility
## Data and preprocessing
## Results (Results Log)
## Investigation of the open dilemma
### Benchmark vs baselines
### Resource profile
### Failure autopsies
### Recommendation
## Artificial Intelligence Usage Disclosure
## Working Notes
### ConvNeXt paper notes (Liu et al., 2022, arXiv:2201.03545)

**Macro design**
- Four stages with a 1:1:3:1 block ratio. ConvNeXt-T uses depths (3, 3, 9, 3) and channels (96, 192, 384, 768), with channels doubling each stage.
- Patchify stem: 4×4 conv, stride 4, non-overlapping, followed by LayerNorm.
- Separate downsampling between stages: LayerNorm, then 2×2 conv with stride 2. Training diverged in the paper without these extra norms.
- Head: global average pool, LayerNorm, linear classifier.

**Block**
1. 7×7 depthwise conv
2. LayerNorm (one per block)
3. 1×1 conv, 4× expansion (inverted bottleneck)
4. GELU (one per block)
5. 1×1 conv back to C channels
6. Layer scale (learnable per-channel, initialised at 1e-6)
7. Stochastic depth, then residual add

**Ablation evidence (ResNet-50 regime, ImageNet-1K top-1, %)**

| Change | Accuracy |
|---|---|
| Enhanced training recipe (baseline) | 78.8 |
| Stage ratio 3:3:9:3 | 79.4 |
| Patchify stem | 79.5 |
| Depthwise conv, widened | 80.5 |
| Inverted bottleneck | 80.6 |
| Depthwise moved up, 7×7 kernel | 80.6 |
| GELU instead of ReLU | 80.6 (unchanged) |
| Fewer activations | 81.3 |
| Fewer norms | 81.4 |
| BatchNorm to LayerNorm | 81.5 |
| Separate downsampling convs | 82.0 |

Kernel-size gains saturated at 7×7 in this regime and at 5×5 in the larger one.

**Paper's training recipe (reference only)**
- AdamW, lr 4e-3, weight decay 0.05, cosine decay, 20 warmup epochs.
- Label smoothing 0.1, layer scale 1e-6, stochastic depth 0.1 for ConvNeXt-T (0.1/0.4/0.5/0.5 for T/S/B/L).
- Mixup, CutMix, RandAugment, Random Erasing and EMA were used on ImageNet. Each needs justifying or dropping for ADNI.

### Limitations and leads from the paper
- **Stated limitation (Appendix F):** ConvNeXt may suit some tasks better than Transformers, e.g. Transformers may be more flexible for multimodal learning or for sparse and structured outputs. Nothing is specific to medical imaging, so the failure analysis relies on my own results.
- **Overfitting:** the paper reports overfitting in larger models on ImageNet-1K and uses EMA to reduce it. ADNI is far smaller, which supports a reduced-size model with stochastic depth, weight decay and possibly EMA.
- **Efficiency:** at equal FLOPs, depthwise convs are known to be slower and more memory-hungry than dense convs, although the paper finds throughput comparable to Swin. To check this, the resource profile compares ConvNeXt latency and VRAM against the CNN, not just parameter counts.
- **Robustness:** the paper tests corrupted and out-of-distribution sets (Appendix B). A small perturbation test (noise, blur, intensity shifts) on ADNI slices could feed the calibration and failure analysis.
- **Scale of evidence:** all reported results are on large natural-image datasets. Design gains may not transfer to small grayscale brain MRI, so the retained-vs-shrunk table states what I kept and why, without assuming the paper's gains carry over.

### Rangpur setup
- Login: `ssh sXXXXXXX@rangpur.compute.eait.uq.edu.au` (UQ network or VPN required). The login node is used for editing, git and job submission only; all computation runs in Slurm jobs.
- Repository cloned over SSH into `$HOME`. Workflow: commit and push locally, then `git pull` on Rangpur.
- Environment: existing conda env `torch`, Python 3.11.15. Versions: torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, matplotlib 3.11.1, pillow 12.3.0, scikit-learn 1.9.1.
- GPU check: `sbatch job.sh` (partition `comp3710`, `--gres=gpu:1`). Result: CUDA available.
- Interactive GPU debugging: `srun --partition=comp3710 --gres=gpu:1 --pty bash`.

### ADNI data audit
- Location: `/home/groups/comp3710/ADNI`. `AD_NC/{train,test}/{AD,NC}` holds 2D JPEG slices (256×240, grayscale). `meta_data_with_label.json` holds 2,189 scans with per-scan labels and file paths.
- Filenames follow `<scanID>_<sliceIndex>.jpeg`. All 1,526 scan IDs in `AD_NC` match JSON keys. The subject ID (`ADNI_<site>_S_<number>`) is parsed from the JSON `raw` path.
- Each scan contributes exactly 20 slices (indices 67 to 114). Provided split: train 1,076 scans (520 AD, 556 NC; 21,520 slices), test 450 scans (223 AD, 227 NC; 9,000 slices).
- 680 subjects with 1 to 8 scans each (282 with a single scan). No subject has both AD and NC scans (221 AD-only, 459 NC-only).
- Leakage: no scan appears in both provided splits, but 216 of 331 test subjects also appear in train. The provided split is not patient-level and is not used as is.
- JSON labels: 0 (NC) 810 scans, 1 (not in the folders) 625 scans, 2 (AD) 754 scans. Folders contain only labels 0 and 2.
- Slice windows: scans fall into 9 distinct 20-slice windows, and window is strongly associated with class. Window 75 to 94 holds 129 AD and 1 NC scan. Windows 94 to 113 and 95 to 114 hold 53 AD and 349 NC scans. A window-only majority classifier reaches 68.2% scan-level accuracy (1,041 of 1,526, fitted and scored on the same data) against 51.3% for always predicting NC. Slice depth is therefore a possible shortcut. Mitigation: report accuracy per window group and compare against a window-only baseline.
- Split plan: pool all scans, stratified 70/15/15 split by subject with a fixed seed, with a test asserting no subject spans two splits.

### Split design
- The provided train/test folders share 216 of 331 test subjects, so they are pooled (1,526 scans, 30,520 slices) and re-split by subject.
- Split unit: subject. All scans and slices of a subject go to one split. No subject has both classes, so class is the stratum.
- Ratios: 70/15/15 train/val/test, seed 42. Validation is used for tuning and early stopping. Test is used only for final results.
- Result: train 476 subjects (155 AD / 321 NC, 1,078 scans, 21,560 slices), val 102 (33 AD / 69 NC, 229 scans, 4,580 slices), test 102 (33 AD / 69 NC, 219 scans, 4,380 slices).
- Slice-level class balance: 49% AD in train, 50% in val, 44% in test. Always predicting NC gives 55.7% on test.
- Window mix is broadly similar across splits, but test has more scans in window 94-113 (26% vs 19% in train), a mostly NC window. Accuracy is reported per window for this reason.
- Limitation: the test set holds 102 subjects (33 AD), so confidence intervals are resampled over subjects, not slices.

### Input size
- All slices are 256 wide × 240 tall, single-channel (8-bit grayscale).
- With a stride-4 stem and three 2× downsamples, 240 rows give 60, 30, 15 and then 7 (the odd 15 drops a row). Padding to 256×256 gives 64, 32, 16, 8.
- Decision: zero-pad 8 rows at top and bottom to 256×256, applied before normalisation so padding matches the black background. No resampling, so no interpolation artefacts.
- Input size is a parameter (`img_size`, default 256) so a smaller size can be tested as an ablation.

### Augmentation and input size
- Slices are brain-extracted (no skull or scalp) and look sagittal, with the cerebellum at the upper left in every scan. The brain's position in the frame varies between scans, and image quality and contrast vary too (see `assets/sample_grid.png`).
- Input: 256×240 slices are zero-padded to 256×256 (8 rows top and bottom) so a stride-4 stem and three 2× downsamples give clean stage sizes of 64, 32, 16 and 8. Padding precedes normalisation and matches the black background. `img_size` is a parameter for ablations.
- Normalisation: mean 0.1086 and std 0.2200, computed on the padded training slices only.
- Augmentation (training only, applied after padding): rotation up to 8°, translation up to 3%, scale 0.95 to 1.05, and brightness and contrast jitter of 0.1. Position varies naturally between scans and contrast varies with scanner and acquisition, so these mimic real variation without cutting off the brain.
- No horizontal or vertical flips. In a sagittal view a left-right flip swaps front and back, producing anatomy that never occurs in the data.
- Candidate failure cases noted from the sample grid: train AD slice 90 (blurry, patchy upper edge) and test NC slice 98 (dark, unusual contrast).

### Smoke test
- Setup: SimpleCNN trained for 3 epochs on a random subset of slices (2,000 per split) as a pipeline check.
- Result: data loading, model, training loop, checkpointing and curve plotting ran end to end without errors. The run's loss, accuracy and timing values were not retained. Accuracy at this scale is not meaningful, so the test only shows that the pipeline is operational.
- Next: the full baseline CNN run (Phase 4) is the first logged result.

### ConvNeXt: retained from the paper vs shrunk

| Component | Paper (ConvNeXt-T) | This project | Why |
|---|---|---|---|
| Patchify stem | 4x4 conv, stride 4 | Retained (1 input channel) | Core design choice; also cuts compute on 256x256 slices |
| Block | 7x7 depthwise conv, LayerNorm, 4x inverted bottleneck, GELU | Retained | Core design |
| Layer scale | init 1e-6 | Retained | Stabilises training of a from-scratch network |
| Stochastic depth | linear rise to 0.1 | Retained, rate set per run | Main regulariser against overfitting 476 training subjects |
| Downsampling | separate LayerNorm + 2x2 stride-2 conv | Retained | Core design |
| Stage layout | 4 stages, depths 3-3-9-3 | 4 stages, depths 2-2-4-2 | Same shape, fewer blocks for ~21k training slices |
| Widths | 96-192-384-768 | 32-64-128-256 | Parameters cut from ~28M to 1.9M to suit the dataset size |
| Training | AdamW, cosine schedule, 300 epochs, batch 4096 | AdamW, cosine schedule with warmup, 30 epochs, batch 64 | Budget of one A100 and ~12 h queue waits |
| Heavy augmentation (RandAugment, Mixup, CutMix, EMA) | Used | Not used | Mixing slices is anatomically unrealistic; light affine and intensity jitter instead |

### Training run log (10 Oct 2026)

All runs use the subject-level split (seed 42), the same augmentation and the script defaults (4 CPUs, 4 data-loading workers, 3 h limit, one A100 on the `comp3710` partition). The course allows one running job per user, so the runs execute one after another.

| Run name | Job ID | Model | Command (after `sbatch --job-name=<name> slurm/train.sh`) |
|---|---|---|---|
| `cnn` | 646768 | CNN | `--model cnn --epochs 30` (constant LR, pipeline check) |
| `resnet_cos` | 646842 | ResNet | `--model resnet --epochs 30 --schedule cosine` |
| `cnn_cos` | 646843 | CNN | `--model cnn --epochs 30 --schedule cosine` |
| `convnext_a` | 646844 | ConvNeXt | `--model convnext --epochs 30 --schedule cosine --lr 1e-3 --weight-decay 0.05 --drop-path 0.1` |
| `convnext_b` | 646845 | ConvNeXt | as A with `--lr 5e-4` |
| `convnext_c` | 646846 | ConvNeXt | as A with `--weight-decay 0.1 --drop-path 0.2 --label-smoothing 0.1` |

The first CNN run (`cnn`) used a constant learning rate and serves as a pipeline check. It trains at about 90 s per epoch, limited by data loading. Its validation accuracy swings between about 0.50 and 0.73 while training accuracy reaches about 0.87, so the cosine-schedule runs form the matched comparison. ConvNeXt B and C each change one thing relative to A (learning rate, regularisation), and the final configuration is chosen on validation accuracy and validation loss only. Parameter counts: CNN about 94k, ResNet 700,146, ConvNeXt 1,899,874.