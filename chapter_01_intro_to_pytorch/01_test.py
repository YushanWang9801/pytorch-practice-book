import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("当前 GPU:", torch.cuda.get_device_name(0))
    print("GPU 数量:", torch.cuda.device_count())
