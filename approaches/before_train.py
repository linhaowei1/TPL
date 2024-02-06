import os
import torch

def prepare(args, model):

    if 'hat' in args.baseline:
        args.mask_pre = None
        args.mask_back = None
        args.reg_lambda = 0.75
        if args.task > 0:
            print('load mask matrix ....')
            args.mask_pre = torch.load(os.path.join(args.prev_output, 'mask_pre'), map_location='cpu')
            args.mask_back = torch.load(os.path.join(args.prev_output, 'mask_back'), map_location='cpu')

            for k, v in args.mask_pre.items():
                args.mask_pre[k] = args.mask_pre[k].cuda()

            for k, v in args.mask_back.items():
                args.mask_back[k] = args.mask_back[k].cuda()
        
            for n, p in model.named_parameters():
                p.grad = None
                if n in args.mask_back.keys():
                    p.hat = args.mask_back[n]
                else:
                    p.hat = None
