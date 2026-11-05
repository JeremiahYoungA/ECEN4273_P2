import json
import random
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor

import yaml

DATASETS = Path(__file__).resolve().parent / "datasets"
NAMES = ["person", "cat", "dog", "dalek", "lightsaber"]
MIN_OBJECT_AREA = 0.25 
IMAGES_PER_CLASS = 500
DOWNLOAD_WORKERS = 16


def download(url, destination):
    """Cache downloads and avoid reusing incomplete files."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        urlretrieve(url, temporary)
        temporary.replace(destination)

def download_img(filename, image_output):
    """Download one image"""
    url = (
        "https://s3.amazonaws.com/images.cocodataset.org/"
        f"train2017/{filename}"
    )
    download(url, image_output / filename)


def prepare_coco():
    """Select images per class for training, verification, and testing"""
    archive = DATASETS / "cache/annotations_trainval2017.zip"
    download(
        "https://s3.amazonaws.com/images.cocodataset.org/"
        "annotations/annotations_trainval2017.zip",
        archive,
    )
    with ZipFile(archive) as zipped:
        annotations = json.loads(zipped.read("annotations/instances_train2017.json"))

    # Original COCO category IDs differ from YOLO class IDs. Map by name.
    # comprehension, just looks odd on different lines
    class_map = {
        category["id"]: NAMES.index(category["name"])
        for category in annotations["categories"]
        if category["name"] in NAMES[:3]
    }
    boxes = defaultdict(list)
    crowd_images = set()
    for annotation in annotations["annotations"]:
        if annotation["category_id"] in class_map:
            boxes[annotation["image_id"]].append(annotation)
            if annotation.get("iscrowd", 0):
                crowd_images.add(annotation["image_id"])

    images = {image["id"]: image for image in annotations["images"]}
    candidates = defaultdict(set)

    for image_id, image_boxes in boxes.items():
        # Keep the existing exclusion for target crowd annotations.
        if image_id in crowd_images:
            continue

        image = images[image_id]
        image_width = image["width"]
        image_height = image["height"]
        image_area = image_width * image_height

        classes_in_image = set()
        reject_image = False

        for box in image_boxes:
            x, y, w, h = box["bbox"]

            # Clip the bounding box to the image boundaries.
            left = max(0, x)
            top = max(0, y)
            right = min(image_width, x + w)
            bottom = min(image_height, y + h)

            # Reject images with invalid target boxes too.
            if right <= left or bottom <= top:
                reject_image = True
                break

            box_area = (right - left) * (bottom - top)

            if box_area < MIN_OBJECT_AREA * image_area:
                reject_image = True
                break

            classes_in_image.add(class_map[box["category_id"]])

        if not reject_image:
            for class_id in classes_in_image:
                candidates[class_id].add(image_id)

    rng = random.Random(76)
    selected = set()

    for class_id in range(3):
        print(f"{NAMES[class_id]} candidates: {len(candidates[class_id])}")

    # Split the combined selection once so a multi-class image cannot leak
    # into two sets. Sorting before shuffling makes the result repeatable.
    selected = sorted(selected)
    rng.shuffle(selected)
    train_end = int(len(selected) * 0.70)
    val_end = train_end + int(len(selected) * 0.15)
    splits = {
        "train": selected[:train_end],
        "val": selected[train_end:val_end],
        "test": selected[val_end:],
    }

    for split, image_ids in splits.items():
        image_output = DATASETS / "coco/images" / split
        label_output = DATASETS / "coco/labels" / split
        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        filenames = [images[image_id]["file_name"] for image_id in image_ids]

        print(f"Checking/downloading {len(filenames)} {split} images...")

        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            results = executor.map(
                download_img,
                filenames,
                repeat(image_output),
            )

            for completed, _ in enumerate(results, start=1):
                if completed % 100 == 0 or completed == len(filenames):
                    print(f"  {split}: {completed}/{len(filenames)} ready")

        for image_id in image_ids:
            image = images[image_id]
            filename = image["file_name"]

            lines = []
            for box in boxes[image_id]:
                x, y, w, h = box["bbox"]  # COCO: top-left x/y and width/height in pixels.
                left, top = max(0, x), max(0, y)
                right, bottom = min(image["width"], x + w), min(image["height"], y + h)
                if right <= left or bottom <= top:
                    continue
                cx = (left + right) / (2 * image["width"])
                cy = (top + bottom) / (2 * image["height"])
                width = (right - left) / image["width"]
                height = (bottom - top) / image["height"]
                lines.append(f"{class_map[box['category_id']]} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
            (label_output / Path(filename).with_suffix(".txt")).write_text("\n".join(lines) + "\n")

        # Use these lists, not folders, to exclude older cached files.
        manifest = DATASETS / "coco" / f"{split}.txt"
        manifest.write_text("".join(f"{image_output / images[i]['file_name']}\n" for i in image_ids))
        print(f"COCO {split}: {len(image_ids)} images")
        for class_id in range(3):
            print(f"  {NAMES[class_id]}: {len(set(image_ids) & candidates[class_id])}")


def main():
    prepare_coco()

    configuration = {
        "path": str(DATASETS),
        "train": ["coco/train.txt"],
        "val": ["coco/val.txt"],
        "test": ["coco/test.txt"],
        "names": dict(enumerate(NAMES)),
    }

    for split in ("train", "val", "test"):
        folder = DATASETS / "custom" / "images" / split

        if folder.is_dir() and any(folder.iterdir()):
            configuration[split].append(f"custom/images/{split}")
        else:
            print(f"No custom {split} images; using COCO only.")

    (DATASETS / "project.yaml").write_text(
        yaml.safe_dump(configuration, sort_keys=False)
    )

    print("Ready: datasets/project.yaml")


if __name__ == "__main__":
    main()
