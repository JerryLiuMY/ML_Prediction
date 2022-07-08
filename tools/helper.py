import psutil
import torch
import nvsmi

# virtual env
# tmux attach -t ml_prediction
# conda activate ml_prediction

# check CUDA
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# check CPU and memory usage
print(psutil.virtual_memory().percent)
print(psutil.cpu_percent())

# check GPU usage
print(list(nvsmi.get_gpus())[0])
print(list(nvsmi.get_gpus())[1])
