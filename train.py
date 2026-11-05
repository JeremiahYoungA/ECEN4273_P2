import os
from pathlib import Path
from ultralytics import YOLO

workspace = Path(os.environ.get("WORKSPACE", os.getcwd()))
build_number = os.environ.get("BUILD_NUMBER", "local")

model = YOLO("architectures/yolo26n-embedded.yaml").load("yolo26n.pt")

model.train(
    data="datasets/project.yaml",
    epochs=1,
    imgsz=320,
    batch=16,
    device=0,
    workers=4,
    cache="ram",
    amp=True,
    project=str(workspace / "runs" / "ci"),
    name=f"build-{build_number}",
    exist_ok=True
)