import numpy as np
import torch
import json
import torch.nn.functional as F
from utils import utils
import torch.nn as nn


@torch.no_grad()
def mds(args, test_logits, test_hidden, loader, task_mask):
    test_samples = torch.Tensor(test_hidden)
    score_in = utils.maha_score(args, test_samples, args.precision_list, args.feat_mean_list, task_mask)
    return score_in

@torch.no_grad()
def mls(args, test_logits, test_hidden, loader, task_mask):
    logits = test_logits
    return torch.tensor(logits).max(-1)[0]

def calculate_mask(self, w):
    contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
    self.thresh = np.percentile(contrib, self.p)
    mask = torch.Tensor((contrib > self.thresh)).cuda()
    self.masked_w = w * mask

def TPLR(args, test_logits, test_hidden, loader, task_mask):

    logit_score = torch.tensor(test_logits)
    logit_score = torch.max(logit_score, dim=1)[0]
    
    test_samples = torch.Tensor(test_hidden)
    p_in = utils.maha_score(args, test_samples, args.precision_list, args.feat_mean_list, task_mask)

    D_out = args.index_out[task_mask].search(utils.normalize(test_samples).astype(np.float32), args.K)
    p_out = torch.tensor(D_out[0][:, -1])


    logit_score = logit_score / args.mls_scale[task_mask]
    p_in = p_in / args.mds_scale[task_mask]
    p_out = p_out / args.knn_scale[task_mask]

    e1 = p_in + p_out
    e2 = logit_score

    composition = -torch.logsumexp(torch.stack((-e1, -e2), dim=0), dim=0)
    
    return composition


def baseline(args, results):

    metric = {}
    ood_label = {}
    ood_score = {}
    sum_ = 0

    for eval_t in range(args.task + 1):

        metric[eval_t] = {}
        ood_label[eval_t] = {}
        ood_score[eval_t] = {}

        logits = np.transpose(results[eval_t]['logits'], (1, 0, 2))     # (task_mask, sample, logit)
        softmax = torch.softmax(torch.from_numpy(logits / 1.0), dim=-1)

        for task_mask in range(args.task+1):

            score_in = None # (samples)
            
            test_logits = results[eval_t]['logits'][task_mask]
            test_hidden = results[eval_t]['hidden'][task_mask]
            loader = args.test_loaders[eval_t]

        
            score_in = TPLR(args, test_logits, test_hidden, loader, task_mask)
            
            # calibration
            score_in = args.calib_b[task_mask] + args.calib_w[task_mask] * score_in

            ood_score[eval_t][task_mask] = score_in.cpu().numpy().tolist()
            ood_label[eval_t][task_mask] = [1] * len(ood_score[eval_t][task_mask]) if eval_t == task_mask else [-1] * len(ood_score[eval_t][task_mask])

        tp_logits = torch.stack([torch.tensor(ood_score[eval_t][task_mask]) for task_mask in range(args.task + 1)], dim=-1) # (sample, task_num)
        tp_softmax = torch.softmax(tp_logits / 0.05, -1)
        task_prediction = torch.max(tp_logits, dim=1)[1]
        
        prediction = (softmax * tp_softmax.unsqueeze(-1)).view(tp_logits.shape[0], -1).max(-1)[1].cpu().numpy().tolist()

        metric[eval_t]['tp_acc'] = utils.acc(task_prediction, np.array(results[eval_t]['references']) // args.class_num)
        metric[eval_t]['acc'] = utils.acc(prediction, results[eval_t]['references'])

        sum_ += metric[eval_t]['acc']

    auc_avg = 0.0
    fpr_avg = 0.0
    aupr_avg = 0.0

    for task_mask in range(args.task + 1):

        ind_score = ood_score[task_mask][task_mask]
        ind_label = ood_label[task_mask][task_mask]
        for eval_t in range(args.task + 1):
            
            if eval_t == task_mask:
                continue
            
            ood_s = ood_score[eval_t][task_mask]
            ood_l = ood_label[eval_t][task_mask]

            predictions = np.array(ind_score + ood_s)
            references = np.array(ind_label + ood_l)

            auc_avg += utils.auroc(predictions, references)
            fpr_avg += utils.fpr95(predictions, references)
            aupr_avg += utils.aupr(predictions, references)

    metric['auroc'] = auc_avg / ((args.task + 1) * args.task)
    metric['fpr@95'] = fpr_avg / ((args.task + 1) * args.task)
    metric['aupr'] = aupr_avg / ((args.task + 1) * args.task)
            
    print("baseline: ", baseline)
    print(metric)
    metric['average'] = sum_ / (args.task+1)
    print(sum_ / (args.task + 1))
    
    import os
    
    with open(os.path.join(args.output_dir, f'{baseline}_results'), 'a') as f:
        f.write(json.dumps(metric) + '\n')
    
    for eval_t in range(args.task + 1):
        utils.write_result_eval(metric[eval_t]['acc'], eval_t, args)

def scaling(args):

    for eval_t in range(args.task + 1):
        mls_score = torch.max(torch.tensor(args.train_logits[eval_t]), dim=1)[0]
        mds_score = utils.maha_score(args, torch.tensor(args.train_hidden[eval_t]), args.precision_list, args.feat_mean_list, eval_t)

        args.mls_scale[eval_t] = mls_score.mean().data
        args.mds_scale[eval_t] = mds_score.mean().data
    
    print(args.mls_scale)
    print(args.mds_scale)


def calibration(args):

    tp_score = []

    for task_mask in range(args.task + 1):

        test_logits = args.logits_dict[task_mask]
        test_hidden = args.features_dict[task_mask]
        loader = args.replay_loader

        score_in = TPLR(args, test_logits, test_hidden, loader, task_mask)
       
        tp_score.append(score_in)
    
    tp_score = torch.stack(tp_score, dim=1)
    tp_label = torch.tensor(args.replay_labels).cuda() // args.class_num


    args.calib_w = args.calib_w.cuda()
    args.calib_b = args.calib_b.cuda()
    tp_score = tp_score.cuda()
    args.calib_w.requires_grad = True
    args.calib_b.requires_grad = True

    optimizer = torch.optim.SGD([args.calib_w, args.calib_b], lr=0.01, momentum=0.8)

    tp_label = torch.tensor(args.replay_labels).cuda() // args.class_num

    for _ in range(100):

        cal_score = tp_score * args.calib_w + args.calib_b
        loss = F.cross_entropy(cal_score, tp_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    args.calib_w = args.calib_w.cpu().detach()
    args.calib_b = args.calib_b.cpu().detach()
    print(args.calib_w)
    print(args.calib_b)