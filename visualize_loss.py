import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
train_loss = torch.load('./checkpoint/train_loss.pt')
valid_loss = torch.load('./checkpoint/valid_loss.pt')
iter = torch.arange(1, len(valid_loss) + 1)
# print(iter)
plt.figure()
plt.plot(iter,train_loss, label='train_loss')
plt.plot(iter,valid_loss, label = 'valid loss')
plt.xlabel('num epoch')
plt.ylabel('Total Loss')
plt.legend()
plt.savefig("train_val_loss.pdf", bbox_inches='tight')
plt.show()
