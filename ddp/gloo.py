import os
import torch as th
import torch.distributed as dist
import sys
import signal
from torchvision import transforms
from torch.multiprocessing import Process
from torchvision.datasets import MNIST


def allreduce(send, recv):
    """Implementation of a ring-reduce."""
    rank = dist.get_rank()
    size = dist.get_world_size()
    send_buff = th.zeros(send.size())
    recv_buff = th.zeros(send.size())
    accum = th.zeros(send.size())
    accum[:] = send[:]
    # th.cuda.synchronize()

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    for i in range(size - 1):
        if i % 2 == 0:
            # Send send_buff
            send_req = dist.isend(send_buff, right)
            dist.recv(recv_buff, left)
            accum[:] += recv[:]
        else:
            # Send recv_buff
            send_req = dist.isend(recv_buff, right)
            dist.recv(send_buff, left)
            accum[:] += send[:]
        send_req.wait()
    # th.cuda.synchronize()
    recv[:] = accum[:]


def run(rank, size):
    """Distributed function to be implemented later."""
    #    t = th.ones(2, 2)
    t = th.rand(2, 2)
    print(rank, " oida ", t)
    # for _ in range(10000000):
    for _ in range(4):
        c = t.clone()
        dist.all_reduce(c, dist.ReduceOp.SUM)
        # allreduce(t, c)
        t.set_(c / size)  # average
    print(t)


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


if __name__ == "__main__":
    # size = 7
    # processes = []
    # for rank in range(size):
    #     p = Process(target=init_processes, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
    try:
        os.environ["MASTER_ADDR"] = "172.24.33.83"
        os.environ["MASTER_PORT"] = "29500"
        # in Docker Interface eth0@if7
        os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        # dist.init_process_group(backend='gloo', rank=0, world_size=2)
        # dist.init_process_group(backend='gloo', rank=1, world_size=2, init_method='tcp://172.24.33.83:29500')
        dist.init_process_group(backend="gloo", rank=1, world_size=3)
        trainset = MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
        testset = MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = th.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=16)
        train_loader = th.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=16)
        train_loader = th.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = th.utils.data.DataLoader(dataset2, **test_kwargs)
        run(2, 3)

    except KeyboardInterrupt or signal.SIGTERM:
        sys.exit(1)
