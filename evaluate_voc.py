import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    targets = {"person", "cat", "dog"}
    class_ids = {
        name: index
        for index, name in model.names.items()
        if name in targets
    }

    if set(class_ids) != targets:
        raise RuntimeError("Model must contain person, cat, and dog")

    # Cache downloaded data in this job's workspace.
    root = Path("datasets/voc2007-evaluation").resolve()
    raw = root / "raw"
    voc = raw / "VOCdevkit/VOC2007"
    split = voc / "ImageSets/Main/test.txt"

    if not split.exists():
        download(
            f"{ASSETS_URL}/VOCtest_06-Nov-2007.zip",
            dir=raw,
            unzip=True,
            delete=False,
            exist_ok=True,
        )

    # Exclude images with difficult target objects: standard YOLO
    # evaluation does not implement VOC's difficult-object ignore rules.
    eligible = []
    for image_id in sorted(split.read_text().split()):
        annotation = ET.parse(
            voc / "Annotations" / f"{image_id}.xml"
        ).getroot()

        has_difficult_target = any(
            obj.findtext("name") in targets
            and obj.findtext("difficult", "0") == "1"
            for obj in annotation.findall("object")
        )

        if not has_difficult_target:
            eligible.append(image_id)

    if not 1 <= args.samples <= len(eligible):
        raise ValueError(f"samples must be between 1 and {len(eligible)}")

    selected = sorted(random.Random(42).sample(eligible, args.samples))

    images = root / "images/val"
    labels = root / "labels/val"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    counts = dict.fromkeys(sorted(targets), 0)
    image_paths = []

    for image_id in selected:
        annotation = ET.parse(
            voc / "Annotations" / f"{image_id}.xml"
        ).getroot()

        width = float(annotation.findtext("size/width"))
        height = float(annotation.findtext("size/height"))
        lines = []

        for obj in annotation.findall("object"):
            name = obj.findtext("name")
            if name not in targets:
                continue

            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin")) - 1
            ymin = float(box.findtext("ymin")) - 1
            xmax = float(box.findtext("xmax")) - 1
            ymax = float(box.findtext("ymax")) - 1

            x = (xmin + xmax) / (2 * width)
            y = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            lines.append(f"{class_ids[name]} {x} {y} {w} {h}")
            counts[name] += 1

        destination = images / f"{image_id}.jpg"
        if not destination.exists():
            shutil.copyfile(
                voc / "JPEGImages" / f"{image_id}.jpg",
                destination,
            )

        (labels / f"{image_id}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        image_paths.append(str(destination))

    if not all(counts.values()):
        raise RuntimeError(f"Sample must contain all three classes: {counts}")

    # An explicit list prevents old cached images entering the sample.
    manifest = root / "evaluation-images.txt"
    manifest.write_text("\n".join(image_paths), encoding="utf-8")

    configuration = root / "data.yaml"
    configuration.write_text(
        yaml.safe_dump({
            "path": str(root),
            "train": str(manifest),
            "val": str(manifest),
            # Keep model IDs rather than renumbering the three classes.
            "names": model.names,
        }),
        encoding="utf-8",
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, output / "evaluation-images.txt")

    print(f"Evaluating {len(selected)} VOC images")
    print(f"Ground-truth objects: {counts}")

    metrics = model.val(
        data=str(configuration),
        split="val",
        classes=sorted(class_ids.values()),
        device="cpu",
        batch=1,
        workers=0,
        imgsz=640,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        per_class[model.names[int(class_id)]] = {
            "precision": float(precision),
            "recall": float(recall),
            "AP50": float(ap50),
            "AP50_95": float(ap),
        }

    summary = {
        "dataset": "VOC2007 test subset",
        "images": len(selected),
        "seed": 42,
        "ground_truth_counts": counts,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": per_class,
    }

    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nVOC subset mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()