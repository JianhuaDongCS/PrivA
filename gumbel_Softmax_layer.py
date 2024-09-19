
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature,device):
    #y = logits + sample_gumbel(logits.size(),device)
    y = logits
    return F.softmax(y / temperature, dim=-1).to(device)


def gumbel_softmax(logits, temperature, args):

    y = gumbel_softmax_sample(logits, temperature, args.device).to(args.device)

    shape = y.size()



    base = torch.zeros(*shape, dtype=torch.float) + ( 1 / (args.attribute_dim) - args.c)
    #print(base)
    base = base.to(args.device)
    output = torch.floor(y / base)
    y_hard = (output / (output + 1e-20)).int()

    y_hard = y_hard.view(*shape)

    y_hard = (y_hard - y).detach() + y
    return y_hard.to(args.device)