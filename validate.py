import argparse
import json
import random
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    targets = {0: "person", 1: "cat", 2: "dog"}

    for class_id, name in targets.items():
        if model.names.get(class_id) != name:
            raise ValueError(
                "Expected model IDs: 0=person, 1=cat, 2=dog"
            )

    root = Path("datasets/coco").resolve()
    source_manifest = root / f"{args.split}.txt"

    # The original manifest contains laptop paths.
    # Rebuild those paths using the Pi's local dataset directory.
    images = []

    for line in source_manifest.read_text().splitlines():
        if line.strip():
            filename = Path(line).name
            images.append(root / "images" / args.split / filename)

    if not 1 <= args.samples <= len(images):
        raise ValueError(
            f"samples must be between 1 and {len(images)}"
        )

    selected = sorted(
        random.Random(42).sample(sorted(images), args.samples)
    )

    counts = dict.fromkeys(targets.values(), 0)

    for image in selected:
        label = (
            root / "labels" / args.split / f"{image.stem}.txt"
        )

        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(
                f"Missing image or label for {image}"
            )

        for line in label.read_text().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                counts[targets[class_id]] += 1

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    manifest = output / "validation-images.txt"
    manifest.write_text(
        "\n".join(str(image) for image in selected) + "\n"
    )

    configuration = output / "data.yaml"
    configuration.write_text(
        yaml.safe_dump({
            "path": str(root),
            "train": str(manifest),
            "val": str(manifest),
            "names": model.names,
        })
    )

    print(f"Validating {len(selected)} COCO {args.split} images")
    print(f"Ground-truth objects: {counts}")

    if args.split == "train":
        print("These are training-set scores, not independent validation.")

    metrics = model.val(
        data=str(configuration),
        split="val",
        classes=list(targets),
        device="cpu",
        batch=1,
        workers=0,
        imgsz=320,
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
        "dataset": "Prepared COCO2017 subset",
        "split": args.split,
        "images": len(selected),
        "seed": 42,
        "ground_truth_counts": counts,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": per_class,
    }

    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"COCO {args.split} mAP@0.5: {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()