import psutil
import torch
import nvsmi

# RCC anaconda
# module load python
# conda env list
# conda create --prefix=/project2/dachxiu/mingyuliu/env --clone base
# source activate /project2/dachxiu/mingyuliu/env

# RCC
# sinteractive --partition=gpu2 --gres=gpu:1 --time=36:00:00

# virtual env
# ssh risklab.chicagobooth.edu -l mingyuliu -p 22
# tmux attach -t ml_prediction
# conda activate ml_prediction

# check CUDA
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU availability: {torch.cuda.is_available()}")
print(f"GPU counts: {torch.cuda.device_count()}")

# check memory and CPU usage
# mpstat -P ALL 1
print(f"CPU usage: {psutil.cpu_percent()}")
print(f"Mem usage: {psutil.virtual_memory().percent}")

# check GPU usage
print(list(nvsmi.get_gpus())[0])
print(list(nvsmi.get_gpus())[1])
