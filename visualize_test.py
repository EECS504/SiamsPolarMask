import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
output = torch.load('result/network_output.pt')
print(output.keys())