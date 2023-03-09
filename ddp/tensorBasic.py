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
        print(i, " TTTTTOIDA ",t)
    print(t)

if __name__ == "__main__":
    main()
