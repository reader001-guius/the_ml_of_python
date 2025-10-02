import torch, numpy, matplotlib
print("torch:", torch.__version__)
print("numpy:", numpy.__version__)
print("matplotlib:", matplotlib.__version__)
print("CUDA 可用?:", torch.cuda.is_available())
