import torch
import subprocess

# 获取当前的CUDA和PyTorch版本
cuda_version = torch.version.cuda.replace('.', '')
torch_version = torch.__version__.split('+')[0].replace('.', '')

# 构造安装命令
install_command = f"pip install --upgrade \"mmcv<2.1.0\" -f https://download.openmmlab.com/mmcv/dist/cu{cuda_version}/torch{torch_version}/index.html"

# 打印并执行安装命令
print("安装命令:", install_command)
subprocess.check_call(install_command, shell=True)
