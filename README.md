# Double-Attention-Network
This is the PyTorch implementation of A^2-Nets: Double Attention Networks, Y Chen et al NIPS 2018

It can be used as an additional block for building models. Right now, the output tensor has shape (B, c_n, H, W). One can
re-construct the original shape (B, c, H, W) with a single line of code in PyTorch.

## Layer architecture
- Two attention steps
![alt text](https://github.com/nguyenvo09/Double-Attention-Network/blob/master/images/model.png)


