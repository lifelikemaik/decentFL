import os
import torch as th
import torch.distributed as dist
from torch.multiprocessing import Process


def main():
    t = th.rand(2, 2)
    print(t)
    for i in range(4):
        c = t.clone()
        print(i, " CCCCCCCCCCCC ", c)
        t.set_(c)
        print(i, " TTTTTOIDA ", t)
    print(t)


if __name__ == "__main__":
    main()


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
