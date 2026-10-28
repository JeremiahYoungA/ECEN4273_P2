import argparse

import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This Python environment cannot access the GPU")

    print("GPU:", torch.cuda.get_device_name(0))
    model = YOLO(args.model)

    target_names = {"person", "cat", "dog"}
    target_ids = [
        class_id
        for class_id, name in model.names.items()
        if name in target_names
    ]
    if not target_ids:
        raise RuntimeError("Model has none of the requested classes")

    cap = cv2.VideoCapture(args.camera)

    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        print("Click the video window and press Q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read webcam frame")
                break

            original = frame.copy()

            result = model.predict(
                frame,
                device=0,
                classes=target_ids,
                conf=0.25,
                imgsz=640,
                verbose=False,
            )[0]

            annotated = result.plot()

            # Preserve the camera's aspect ratio.
            panel_width = 640
            panel_height = round(
                frame.shape[0] * panel_width / frame.shape[1]
            )

            left = cv2.resize(original, (panel_width, panel_height))
            right = cv2.resize(annotated, (panel_width, panel_height))

            cv2.putText(
                left, "Original", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.putText(
                right, "Detection", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            cv2.imshow(
                "Original | Detection",
                cv2.hconcat([left, right]),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()