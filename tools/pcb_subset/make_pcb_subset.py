#!/usr/bin/env python3
"""Build the PCB tiny-defect subset used in the SSP-DETR paper from PKU-Market-PCB.

Processing:
  1. Downsample every image by a factor of two along each side.
  2. Keep the images that contain at least one defect box whose area is below
     32 x 32 pixels after downsampling (the COCO small-object range).
     All boxes of a kept image are kept, including larger ones.
  3. Assign the kept images to train/test with the fixed lists in splits/.

On the original release (693 images, 2,953 boxes) this keeps 493 images and
2,079 boxes (train 394 / 1,668, test 99 / 411).

Usage:
  python make_pcb_subset.py --src /path/to/PCB_DATASET --out /path/to/pcb_coco
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

from PIL import Image

SCALE = 0.5
SMALL_AREA = 32 * 32
JPEG_QUALITY = 90
CLASS_DIRS = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
CATEGORIES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SPLITS = ("train", "test")


def parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (int(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((obj.find("name").text.lower(), x1, y1, x2, y2))
    return width, height, boxes


def downsample(width, height, boxes):
    # Python's round() (round half to even) reproduces the released image sizes.
    size = (round(width * SCALE), round(height * SCALE))
    # VOC corners are 1-based; COCO boxes are 0-based [x, y, w, h].
    scaled = [(name, [(x1 - 1) * SCALE, (y1 - 1) * SCALE, (x2 - x1) * SCALE, (y2 - y1) * SCALE])
              for name, x1, y1, x2, y2 in boxes]
    return size, scaled


def load_dataset(src):
    records = {}
    for class_dir in CLASS_DIRS:
        ann_dir = os.path.join(src, "Annotations", class_dir)
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith(".xml"):
                continue
            stem = fname[:-4]
            width, height, boxes = parse_voc(os.path.join(ann_dir, fname))
            size, scaled = downsample(width, height, boxes)
            records[stem + ".jpg"] = {
                "image": os.path.join(src, "images", class_dir, stem + ".jpg"),
                "size": size,
                "boxes": scaled,
            }
    return records


def is_selected(record):
    return any(w * h < SMALL_AREA for _, (_, _, w, h) in record["boxes"])


def read_split(split_dir, split):
    with open(os.path.join(split_dir, split + ".txt")) as f:
        return [line.strip() for line in f if line.strip()]


def to_coco(names, records, zero_based):
    offset = 0 if zero_based else 1
    cat_id = {name: i + offset for i, name in enumerate(CATEGORIES)}
    images, annotations = [], []
    for image_id, name in enumerate(names, start=1):
        rec = records[name]
        images.append({"id": image_id, "file_name": name, "width": rec["size"][0], "height": rec["size"][1]})
        for cls, bbox in rec["boxes"]:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": cat_id[cls],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": [],
            })
    categories = [{"id": cat_id[n], "name": n, "supercategory": "pcb_defect"} for n in CATEGORIES]
    info = {"description": "PKU-Market-PCB tiny-defect subset (x0.5 downsampling, >=1 box below 32x32)"}
    return {"info": info, "images": images, "annotations": annotations, "categories": categories}


def resize_image(src_path, size):
    lanczos = getattr(Image, "Resampling", Image).LANCZOS
    return Image.open(src_path).convert("RGB").resize(size, lanczos)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="PKU-Market-PCB root containing Annotations/ and images/")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--splits", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits"),
                        help="directory with train.txt and test.txt")
    parser.add_argument("--no-images", action="store_true", help="write annotations only")
    args = parser.parse_args()

    records = load_dataset(args.src)
    selected = {name for name, rec in records.items() if is_selected(rec)}
    splits = {s: read_split(args.splits, s) for s in SPLITS}
    listed = set(splits["train"]) | set(splits["test"])
    if set(splits["train"]) & set(splits["test"]):
        raise SystemExit("train and test lists overlap")
    if listed != selected:
        raise SystemExit(f"split lists ({len(listed)} images) differ from the selection rule "
                         f"({len(selected)} images): {sorted(listed ^ selected)[:10]}")
    print(f"{len(records)} source images, {len(selected)} selected")

    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    for split in SPLITS:
        for zero_based, suffix in ((False, ""), (True, "_0based")):
            coco = to_coco(splits[split], records, zero_based)
            path = os.path.join(args.out, "annotations", f"instances_{split}{suffix}.json")
            with open(path, "w") as f:
                json.dump(coco, f)
        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} boxes")
        if args.no_images:
            continue
        img_dir = os.path.join(args.out, "images", split)
        os.makedirs(img_dir, exist_ok=True)
        for name in splits[split]:
            rec = records[name]
            resize_image(rec["image"], rec["size"]).save(os.path.join(img_dir, name), quality=JPEG_QUALITY)


if __name__ == "__main__":
    main()
