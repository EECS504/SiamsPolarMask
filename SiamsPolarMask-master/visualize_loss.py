import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
train_loss = torch.load('checkpoint/train_loss.pt')
print((train_loss))
valid_loss = torch.load('checkpoint/valid_loss.pt')
print((valid_loss))
iter = torch.arange(1, len(valid_loss['centerness_loss']) + 1)
print((iter))
iter =iter.numpy()
# print(iter)
plt.figure()
plt.plot(iter,train_loss['centerness_loss'], label='train_loss')
plt.plot(iter,valid_loss['centerness_loss'], label = 'valid loss')
plt.xlabel('num epoch')
plt.ylabel('centerness Loss')
plt.legend()
plt.savefig("train_val_loss.pdf", bbox_inches='tight')
plt.show()
