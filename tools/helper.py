import psutil
import torch
import nvsmi

# virtual env
# tmux attach -t ml_prediction
# conda activate ml_prediction

# check CUDA
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU availability: {torch.cuda.is_available()}")
print(f"GPU counts: {torch.cuda.device_count()}")

# check CPU and memory usage
print(f"Mem usage: {psutil.virtual_memory().percent}")
print(f"CPU usage: {psutil.cpu_percent()}")

# check GPU usage
print(list(nvsmi.get_gpus())[0])
print(list(nvsmi.get_gpus())[1])
