from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import os
import shutil
import torch
import json
import numpy as np
import logging
import math
from transformers import get_scheduler
from utils import utils, baseline
import faiss
logger = logging.getLogger(__name__)

class Appr(object):
    
    def __init__(self, args):
        super().__init__()
        self.args = args

    def eval(self, model, train_loaders, test_loaders, replay_loader, accelerator):
        
        model = accelerator.prepare(model)
        model.eval()
        replay_loader = accelerator.prepare(replay_loader)
        train_hidden, train_labels = None, None
        if os.path.exists(os.path.join(self.args.output_dir, 'train_hidden')):
            with open(os.path.join(self.args.output_dir, 'results'), 'r') as f:
                results = json.load(f)
                results = {int(k): v for (k,v) in results.items()}
            with open(os.path.join(self.args.output_dir, 'train_hidden'), 'r') as f:
                train_hidden = json.load(f)
                train_hidden = {int(k): v for (k,v) in train_hidden.items()}
            with open(os.path.join(self.args.output_dir, 'train_labels'), 'r') as f:
                train_labels = json.load(f)
                train_labels = {int(k): v for (k,v) in train_labels.items()}
            with open(os.path.join(self.args.output_dir, 'train_logits'), 'r') as f:
                train_logits = json.load(f)
                train_logits = {int(k): v for (k,v) in train_logits.items()}
        else:
            results = {}
            train_hidden = {}
            train_labels = {}
            train_logits = {}
            model.eval()

            for eval_t in tqdm(range(self.args.task + 1)):

                results[eval_t] = {
                    'predictions': [],      # [N x data], prediction of N task mask
                    'references': [],       # [data]
                    'hidden': [],           # [N x data]
                    'logits': [],    # [N x data]
                    'softmax_prob': [],     # [N x data]
                    'total_num': 0
                }
                train_hidden[eval_t] = []
                train_labels[eval_t] = []
                train_logits[eval_t] = []
                test_loader, train_loader = accelerator.prepare(test_loaders[eval_t], train_loaders[eval_t])

                for task_mask in range(self.args.task + 1):
                    
                    train_hidden_list = []
                    hidden_list = []
                    prediction_list = []
                    logits_list = []
                    softmax_list = []
                    train_logits_list = []

                    for _, batch in enumerate(test_loader):
                        with torch.no_grad():
                            features, _ = model.forward_features(task_mask, batch[0], s=self.args.smax)
                            output = model.forward_classifier(task_mask, features)
                            output = output[:, task_mask * self.args.class_num: (task_mask+1) * self.args.class_num]
                            score, prediction = torch.max(torch.softmax(output, dim=1), dim=1)

                            hidden_list += (features).cpu().numpy().tolist()
                            prediction_list += (prediction + self.args.class_num * task_mask).cpu().numpy().tolist()
                            softmax_list += score.cpu().numpy().tolist()
                            logits_list += output.cpu().numpy().tolist()

                            if task_mask == 0:
                                results[eval_t]['total_num'] += batch[0].shape[0]
                                results[eval_t]['references'] += batch[1].cpu().numpy().tolist()
                    
                    results[eval_t]['hidden'].append(hidden_list)
                    results[eval_t]['predictions'].append(prediction_list)
                    results[eval_t]['softmax_prob'].append(softmax_list)
                    results[eval_t]['logits'].append(logits_list)

                    
                for _, batch in enumerate(train_loader):
                    with torch.no_grad():
                        features, _ = model.forward_features(eval_t, batch[0], s=self.args.smax)
                        output = model.forward_classifier(eval_t, features)
                        output = output[:, eval_t * self.args.class_num: (eval_t+1) * self.args.class_num]
                        train_logits[eval_t] += output.cpu().numpy().tolist()
                        train_hidden[eval_t] += (features).cpu().numpy().tolist()
                        train_labels[eval_t] += (batch[1] - self.args.class_num * eval_t).cpu().numpy().tolist()

            with open(os.path.join(self.args.output_dir, 'results'), 'w') as f:
                json.dump(results, f)
            with open(os.path.join(self.args.output_dir, 'train_hidden'), 'w') as f:
                json.dump(train_hidden, f)
            with open(os.path.join(self.args.output_dir, 'train_labels'), 'w') as f:
                json.dump(train_labels, f)
            with open(os.path.join(self.args.output_dir, 'train_logits'), 'w') as f:
                json.dump(train_logits, f)

        out_features = {task_mask: [] for task_mask in range(self.args.task + 1)}
        in_features = {task_mask: [] for task_mask in range(self.args.task + 1)}
        features_dict = {task_mask: [] for task_mask in range(self.args.task + 1)}
        logits_dict = {task_mask: [] for task_mask in range(self.args.task + 1)}
        replay_labels = []

        for idx, batch in enumerate(replay_loader):

            with torch.no_grad():
                for task_mask in range(self.args.task + 1):
                    if idx == task_mask: 
                        features, _ = model.forward_features(task_mask, batch[0], s=self.args.smax)
                        in_features[task_mask] += (features).cpu().numpy().tolist()
                    else:
                        features, _ = model.forward_features(task_mask, batch[0], s=self.args.smax)
                        out_features[task_mask] += (features).cpu().numpy().tolist()
                    logits = model.forward_classifier(task_mask, features)[:, task_mask * self.args.class_num: (task_mask+1) * self.args.class_num]
                    features_dict[task_mask] += features.cpu().numpy().tolist()
                    logits_dict[task_mask] += logits.cpu().numpy().tolist()

            replay_labels += batch[1].cpu().numpy().tolist()

        ## replay data
        self.args.out_features = out_features
        self.args.in_features = in_features
        self.args.features_dict = features_dict
        self.args.logits_dict = logits_dict
        self.args.replay_loader = replay_loader
        self.args.replay_labels = replay_labels

        ## train data
        self.args.train_logits = train_logits
        self.args.train_labels = train_labels
        self.args.train_hidden = train_hidden
        self.args.model = model
        self.args.test_loaders = test_loaders

        ## maha feat
        self.args.feat_mean_list, self.args.precision_list = utils.load_maha(self.args, train_hidden, train_labels)
        
        self.args.calib_w = torch.ones(self.args.task + 1)
        self.args.calib_b = torch.zeros(self.args.task + 1)
        self.args.mls_scale = [1.0 for _ in range(self.args.task + 1)]
        self.args.mds_scale = [1.0 for _ in range(self.args.task + 1)]
        self.args.knn_scale = [1.0 for _ in range(self.args.task + 1)]
        self.args.index_out = [None for _ in range(self.args.task + 1)]
        self.args.tplr_setup = [False for _ in range(self.args.task + 1)]

        for task_mask in range(self.args.task + 1):
            self.args.index_out[task_mask] = faiss.IndexFlatL2(len(self.args.out_features[task_mask][0]))
            self.args.index_out[task_mask].add(utils.normalize(self.args.out_features[task_mask]).astype(np.float32))
            self.args.tplr_setup[task_mask] = True

        baseline.scaling(self.args)
        baseline.calibration(self.args)
        baseline.baseline(self.args, results)