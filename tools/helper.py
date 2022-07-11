import psutil
import torch
import nvsmi

# RCC
# ssh midway2.rcc.uchicago.edu -l mingyuliu -p 22
# tmux attach -t ml_prediction
# sinteractive --partition=gpu2 --ntasks=50 --nodes=2 --gres=gpu:2 --time=36:00:00
# module load python/anaconda-2020.02
# module load cuda/11.2
# conda activate base
# cd /project2/dachxiu/mingyuliu

# Risklab
# ssh risklab.chicagobooth.edu -l mingyuliu -p 22
# tmux attach -t ml_prediction
# conda activate ml_prediction
# cd /project2/mingyuliu

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

# RCC setup
# ssh midway2.rcc.uchicago.edu -l mingyuliu -p 22
# tmux attach -t ml_prediction
# module load python
# module load cuda/11.2
# conda activate base
# pip install -r requirements.txt
# pip install "mxnet_cu101<2.0.0, >=1.7.0"
# pip install lightgbm --install=--gpu

# Risklab setup
# ssh risklab.chicagobooth.edu -l mingyuliu -p 22
# tmux attach -t ml_prediction
# conda activate ml_prediction
# conda install -c anaconda cudatoolkit=10.2
# pip install -r requirements.txt
# pip install "mxnet_cu101<2.0.0, >=1.7.0"
# pip install lightgbm --install=--gpu
