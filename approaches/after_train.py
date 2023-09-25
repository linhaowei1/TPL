import os
import torch
from utils.sgd_hat import cum_mask, freeze_mask    
def compute(args, model):
    args.mask_pre = cum_mask(smax=args.smax, t=args.task, model=model, mask_pre=args.mask_pre)
    args.mask_back = freeze_mask(args.smax, args.task, model, args.mask_pre)
    torch.save(args.mask_pre, os.path.join(args.output_dir, 'mask_pre'))
    torch.save(args.mask_back, os.path.join(args.output_dir, 'mask_back'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model'))

