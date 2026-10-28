import torch
import ultralytics

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"

print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((100, 100), device="cuda")
print("GPU calculation:", (x @ x).sum().item())
torch.cuda.synchronize()
print("GPU check passed")