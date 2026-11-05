"""Benchmark raw NCNN inference; no YOLO postprocessing or accuracy metrics."""

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image, ImageOps


def prepare_image(path, size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(size / image.width, size / image.height)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(image, ((size - width) // 2, (size - height) // 2))
        pixels = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "vulkan"), default="vulkan")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    params = sorted(args.model.glob("*.param"))
    if len(params) != 1 or not params[0].with_suffix(".bin").is_file():
        raise RuntimeError("Expected exactly one .param file and its matching .bin file")
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and p.is_file())
    if not 1 <= args.samples <= len(images):
        raise ValueError(f"Requested {args.samples} images, but found {len(images)}")
    images = sorted(random.Random(42).sample(images, args.samples))

    net = ncnn.Net()
    if args.device == "vulkan":
        net.opt.use_vulkan_compute = True
        print(f"Selecting Vulkan device {args.device_index}; verify it is Adreno in the NCNN device list.", flush=True)
        net.set_vulkan_device(args.device_index)
    start = time.perf_counter()
    if net.load_param(str(params[0])) != 0:
        raise RuntimeError("NCNN failed to load model parameters")
    if net.load_model(str(params[0].with_suffix('.bin'))) != 0:
        raise RuntimeError("NCNN failed to load model weights")
    load_seconds = time.perf_counter() - start
    inputs, outputs = list(net.input_names()), sorted(net.output_names())
    if len(inputs) != 1 or not outputs:
        raise RuntimeError(f"Expected one input and at least one output; got {inputs}, {outputs}")
    print(f"Model loaded in {load_seconds:.2f} s; input={inputs}, outputs={outputs}", flush=True)

    def run(image):
        extractor = net.create_extractor()
        if extractor.input(inputs[0], ncnn.Mat(image)) != 0:
            raise RuntimeError("NCNN rejected the input")
        arrays = []
        for name in outputs:
            code, result = extractor.extract(name)
            if code != 0:
                raise RuntimeError(f"NCNN extraction failed for {name}: {code}")
            arrays.append(np.array(result))
        return arrays

    warmup = run(prepare_image(images[0], 320))
    if not all(a.size and np.isfinite(a).all() for a in warmup):
        raise RuntimeError("Model output is empty or contains non-finite values")
    print(f"Warmup output shapes: {[a.shape for a in warmup]}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(images, 1):
        start = time.perf_counter()
        image = prepare_image(path, 320)
        prepared = time.perf_counter()
        result = run(image)
        finished = time.perf_counter()
        if not all(a.size and np.isfinite(a).all() for a in result):
            raise RuntimeError(f"Invalid output for {path}")
        rows.append((str(path), (prepared - start) * 1000, (finished - prepared) * 1000))
        print(f"{index}/{len(images)} {path.name}: inference {rows[-1][2]:.1f} ms", flush=True)

    with (args.output / "timings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("image", "preprocess_ms", "inference_ms"))
        writer.writerows(rows)
    mean_ms = statistics.mean(row[2] for row in rows)
    summary = {"device": args.device, "device_index": args.device_index, "images": len(rows), "seed": 42, "image_size": 320, "load_seconds": load_seconds, "mean_preprocess_ms": statistics.mean(row[1] for row in rows), "mean_inference_ms": mean_ms, "median_inference_ms": statistics.median(row[2] for row in rows), "inference_fps": 1000 / mean_ms, "note": "Raw model output only; no detection postprocessing or accuracy metrics."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
