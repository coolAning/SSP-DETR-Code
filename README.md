# SSP-DETR: A Shared Spatial Prior for Tiny Defect Detection in Electronics Manufacturing

This repository provides the implementation of SSP-DETR. SSP-DETR learns a normalized spatial defect prior from high-resolution encoder memory with box supervision. The prior is shared between encoder memory reweighting, through Prior Conditioned Spatial Gating (PCSG), and final-layer Hungarian assignment, through Spatial Prior Matching.

## Installation

Python 3.9 and CUDA are recommended.

```bash
conda create -n dqdetr python=3.9 -y
conda activate dqdetr
bash install.sh
```

Verify the compiled multi-scale deformable-attention operator:

```bash
cd models/dqdetr/ops
python test.py
```

## Data

CPH2D and PCB use COCO-style layouts:

```text
dataset_root/
|-- annotations/
|   |-- instances_train.json
|   `-- instances_val.json
|-- train/
`-- val/
```

Equivalent `images/train`, `images/val`, `test`, and `instances_test.json` layouts are also supported. CPS2D uses `train_images`, `test_images`, `annotations/train_labels.json`, and `annotations/test_labels.json`.

### Dataset Availability

| Dataset | Location |
| --- | --- |
| PCB | [Hugging Face dataset](https://huggingface.co/datasets/awesome-aning/SSP-DETR-PCB) |
| CPS2D | To be released. |
| CPH2D | To be released under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. |

The PCB benchmark is a tiny-defect subset of PKU-Market-PCB. It can be rebuilt from the original release with `tools/pcb_subset/make_pcb_subset.py`; see [tools/pcb_subset/README.md](tools/pcb_subset/README.md) for the selection rule.

## Model Checkpoints

The model weights are hosted separately from the datasets. Each dataset has three checkpoints, one for each training seed (42, 43, and 44) reported in the paper. All three runs of a dataset use the same configuration: `config/DQ_5scale_<dataset>_fixed900_pmap_ramp8.py`. In configuration names, checkpoint names, and the `DATASET` setting of the scripts, the datasets are written as `pcb`, `cps2d`, and `cph` (CPH2D).

| Dataset | Seed | AP | AP75 | APS | Model checkpoint |
| --- | ---: | ---: | ---: | ---: | --- |
| PCB | 42 | 57.04 | 59.31 | 56.60 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_pcb_seed42.pth) |
| PCB | 43 | 57.49 | 64.11 | 57.15 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_pcb_seed43.pth) |
| PCB | 44 | 56.51 | 57.46 | 55.99 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_pcb_seed44.pth) |
| PCB | mean ± SD | 57.0 ± 0.5 | 60.3 ± 3.4 | 56.6 ± 0.6 | |
| CPS2D | 42 | 23.14 | 17.99 | 19.89 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_cps2d_seed42.pth) |
| CPS2D | 43 | 24.18 | 19.05 | 20.91 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_cps2d_seed43.pth) |
| CPS2D | 44 | 23.38 | 18.61 | 20.21 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_cps2d_seed44.pth) |
| CPS2D | mean ± SD | 23.6 ± 0.5 | 18.6 ± 0.5 | 20.3 ± 0.5 | |
| CPH2D | 42 | 31.27 | 24.81 | 27.98 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_cph_seed42.pth) |
| CPH2D | 43 | 29.70 | 32.21 | 23.84 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_cph_seed43.pth) |
| CPH2D | 44 | 32.93 | 34.63 | 25.95 | [download](https://huggingface.co/awesome-aning/SSP-DETR/resolve/main/ssp_detr_cph_seed44.pth) |
| CPH2D | mean ± SD | 31.3 ± 1.6 | 30.6 ± 5.1 | 25.9 ± 2.1 | |

The per-seed values are the outputs of `scripts/eval.sh` and match the per-seed results in the supplementary material; the mean ± SD rows (sample standard deviation over the three seeds) match Table II of the paper. The checkpoints contain only the model state dictionary under the `model` key.

## Evaluation

Edit `DATASET`, `SEED`, `DATASET_ROOT`, and `GPU_ID` near the top of `scripts/eval.sh`, then run:

```bash
bash scripts/eval.sh
```

The same values can also be set in the environment, for example:

```bash
DATASET=pcb SEED=43 DATASET_ROOT=/path/to/pcb bash scripts/eval.sh
```

The script selects the matching configuration and checkpoint, downloads the checkpoint when needed, verifies its SHA256 checksum, and runs COCO-style evaluation on one GPU.

## Training

Edit the dataset, pretrained model, output directory, seed, and GPU settings near the top of `scripts/train.sh`, then run:

```bash
bash scripts/train.sh
```

The paper reports one run each with `SEED` set to 42, 43, and 44, trained on two GPUs (the default `GPU_IDS="0,1"`).

## Acknowledgement

This implementation is based on the public DQ-DETR and DINO codebases. Source-level attribution is retained in the relevant files.

```bibtex
@inproceedings{huang2024dq,
  title={DQ-DETR: DETR with Dynamic Query for Tiny Object Detection},
  author={Huang, Yi-Xin and Liu, Hou-I and Shuai, Hong-Han and Cheng, Wen-Huang},
  booktitle={European Conference on Computer Vision},
  pages={290--305},
  year={2025},
  organization={Springer}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). The datasets are distributed separately under their own terms; see [Dataset Availability](#dataset-availability).
