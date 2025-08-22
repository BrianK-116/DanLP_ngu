# check_gpu.py
import torch
print(f"PyTorch CUDA version: {torch.version.cuda}")

# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()

print(f"Is CUDA available? -> {is_cuda_available}")

if is_cuda_available:
    # Get the number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    
    # Get the name of the current GPU
    current_gpu_name = torch.cuda.get_device_name(0)
    print(f"Current GPU Name: {current_gpu_name}")
else:
    print("PyTorch cannot find a CUDA-enabled GPU.")
    print("Please review the installation steps.")