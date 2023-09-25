import torch
from torch.optim.optimizer import required
from torch.optim import SGD
import numpy as np

class SGD_hat(SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False):
        super(SGD_hat, self).__init__(params, lr, momentum, dampening,
                                      weight_decay, nesterov)

    @torch.no_grad()
    def step(self, closure=None, hat=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                if hat:
                    if p.hat is not None:
                        d_p = d_p * p.hat

                p.add_(d_p, alpha=-group['lr'])

        return loss

def HAT_reg(args, masks):
    """ masks and self.mask_pre must have values in the same order """
    reg, count = 0., 0.
    if args.mask_pre is not None:
        for m, mp in zip(masks, args.mask_pre.values()):
            aux = 1. - mp
            reg += (m * aux).sum()
            count += aux.sum()
    else:
        for m in masks:
            reg += m.sum()
            count += np.prod(m.size()).item()
    reg /= count
    return args.reg_lambda * reg

def compensation(model, args, thres_cosh=50, s=1):
    """ Equation before Eq. (4) in the paper """
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad *= args.smax / s * num / den

def compensation_clamp(model, thres_emb=6):
    # Constrain embeddings
    for n, p in model.named_parameters():
        if 'ec' in n:
            if p.grad is not None:
                p.data.copy_(torch.clamp(p.data, -thres_emb, thres_emb))

def cum_mask(smax, t, model, mask_pre):
    """ 
        Keep track of mask values. 
        This will be used later as a regularizer in the optimization
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    task_id = torch.tensor([t]).cuda()
    mask = {}
    for n, _ in model.named_parameters():
        names = n.split('.')
        checker = [i for i in ['ec0', 'ec1', 'ec2'] if i in names]
        if names[0] == 'module':
            names = names[1:]
        if checker:
            if 'adapter' in n:
                gc1, gc2 = model.__getattr__(names[0])[int(names[1])].__getattr__(names[2]).mask(task_id, s=smax)
                if checker[0] == 'ec1':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc1.detach()
                    mask[n].requires_grad = False
                elif checker[0] == 'ec2':
                    n = '.'.join(n.split('.')[:-1])
                    mask[n] = gc2.detach()
                    mask[n].requires_grad = False

    if mask_pre is None:
        mask_pre = {}
        for n in mask.keys():
            mask_pre[n] = mask[n]
    else:
        for n in mask.keys():
            mask_pre[n] = torch.max(mask_pre[n], mask[n])
    return mask_pre

def freeze_mask(P, t, model, mask_pre):
    """
        Eq (2) in the paper. self.mask_back is a dictionary whose keys are
        the convolutions' parameter names. Each value of a key is a matrix, whose elements are
        approximately binary.
    """
    try:
        model = model.module
    except AttributeError:
        model = model

    mask_back = {}
    for n, p in model.named_parameters():
        names = n.split('.')
        if 'adapter' in n: # adapter1 or adapter2. adapter.ec1, adapter.ec2
            # e.g. n is blocks.1.adapter1.fc1.weight
            if 'fc1.weight' in n:
                mask_back[n] = 1 - mask_pre['.'.join(names[:-2]) + '.ec1'].data.view(-1, 1).expand_as(p)
            elif 'fc1.bias' in n:
                mask_back[n] = 1 - mask_pre['.'.join(names[:-2]) + '.ec1'].data.view(-1)
            elif 'fc2.weight' in n:
                post = mask_pre['.'.join(names[:-2]) + '.ec2'].data.view(-1, 1).expand_as(p)
                pre  = mask_pre['.'.join(names[:-2]) + '.ec1'].data.view(1, -1).expand_as(p)
                mask_back[n] = 1 - torch.min(post, pre)
            elif 'fc2.bias' in n:
                mask_back[n] = 1 - mask_pre['.'.join(names[:-2]) + '.ec2'].view(-1)
    return mask_back
    