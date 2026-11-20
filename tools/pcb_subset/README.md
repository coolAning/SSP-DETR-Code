# PCB Tiny-Defect Subset

`make_pcb_subset.py` rebuilds the PCB benchmark from the original PKU-Market-PCB release (W. Huang and P. Wei, "A PCB dataset for defects detection and classification," arXiv:1901.08204, 2019). The output is identical to the [Hugging Face PCB dataset](https://huggingface.co/datasets/awesome-aning/SSP-DETR-PCB).

## Processing

1. Every image is downsampled by a factor of two along each side (Pillow Lanczos resampling, JPEG quality 90).
2. An image is kept if at least one of its defect boxes has an area below 32 × 32 pixels after downsampling, which is the COCO small-object range. All boxes in a kept image are kept, including larger ones.
3. Kept images are assigned to the training and test sets with the fixed lists in `splits/`.

| | Images | Boxes |
| --- | ---: | ---: |
| PKU-Market-PCB (original) | 693 | 2,953 |
| Subset, train | 394 | 1,668 |
| Subset, test | 99 | 411 |
| Subset, total | 493 | 2,079 |

Boxes are converted from 1-based VOC corners to 0-based COCO `[x, y, w, h]` and scaled by 0.5. The six categories are `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`, with IDs 1–6 in `instances_*.json` and 0–5 in `instances_*_0based.json`.

## Usage

```bash
pip install pillow
python tools/pcb_subset/make_pcb_subset.py --src /path/to/PCB_DATASET --out /path/to/pcb_coco
```

`--src` is the original release root, which contains `Annotations/<Class>/*.xml` and `images/<Class>/*.jpg`. The script checks that the selection rule picks exactly the images listed in `splits/` and stops otherwise. Add `--no-images` to write only the annotations.

Output:

```text
pcb_coco/
|-- annotations/
|   |-- instances_{train,test}.json
|   `-- instances_{train,test}_0based.json
`-- images/
    |-- train/
    `-- test/
```

The annotations are identical to the released ones. With Pillow 9.3 the images are also pixel-identical; other Pillow or libjpeg versions may produce slightly different JPEG pixels.
