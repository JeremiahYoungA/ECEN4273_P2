from ultralytics import YOLO

model = YOLO("models/best.pt")
model.to("cpu")

required = {"person", "cat", "dog"}
available = set(model.names.values())
missing = required - available

if missing:
    raise RuntimeError(f"Model is missing required classes: {missing}")

print("Model loaded successfully on Pi CPU")
print("Required classes found: person, cat, dog")
print("Total model classes:", len(model.names))