# Double-Attention-Network
This is the PyTorch implementation of Double Attention Network, NIPS 2018

It can be used as an additional block for building models. Right now, the output tensor has shape (B, c_n, H, W). One can
re-construct the original shape (B, c, H, W) with a single line of code in PyTorch.

## Layer architecture
- Temporal behavior of guardians
![alt text](https://github.com/nguyenvo09/Double-Attention-Network/blob/master/images/model.png)


