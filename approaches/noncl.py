from modulefinder import IMPORT_NAME
from approaches import before_train, after_train
from tqdm.auto import tqdm
import torch.nn as nn
import os
import shutil
import torch
import numpy as np
import logging
import math
from transformers import get_scheduler
from utils.sgd_hat import HAT_reg, compensation, compensation_clamp
from utils.sgd_hat import SGD_hat as SGD
from utils import utils
logger = logging.getLogger(__name__)


class Appr(object):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train(self, model, train_loader, test_loaders, replay_loader):

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        optimizer = SGD(model.adapter_parameters(), lr=self.args.learning_rate,
                        momentum=0.9, weight_decay=5e-4, nesterov=True)
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        model = model.cuda()
       

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format(len(train_loader) * self.args.batch_size))
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}, checkpoint Model = {self.args.model_name_or_path}")
        logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Learning Rate = {self.args.learning_rate}")
        logger.info(f"  Seq ID = {self.args.idrandom}, Task id = {self.args.task}, Task Name = {self.args.task_name}, Num task = {self.args.ntasks}")

        progress_bar = tqdm(range(self.args.max_train_steps))
        completed_steps = 0
        starting_epoch = 0

        for epoch in range(starting_epoch, self.args.num_train_epochs):
            model.train()

            for step, batch in enumerate(train_loader):

                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()

                
                outputs = model(batch[0])
                
                loss = nn.functional.cross_entropy(outputs, batch[1])

                loss.backward()

                if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while
            
            if completed_steps >= self.args.max_train_steps:
                break
        

        for eval_t in range(self.args.task + 1):
            results = self.eval_cil(model, test_loaders, eval_t)
            print("*task {}, til_acc = {}, cil_acc = {}, tp_acc = {}".format(
                eval_t, results['til_accuracy'], results['cil_accuracy'], results['TP_accuracy']))
            utils.write_result(results, eval_t, self.args)

    def eval_cil(self, model, test_loaders, eval_t):
        model.eval()
        dataloader = test_loaders[eval_t]
        label_list = []
        cil_prediction_list, til_prediction_list = [], []
        total_num = 0

        for _, batch in enumerate(dataloader):
            with torch.no_grad():
                
                features = model.forward_features(batch[0].cuda())
                logits = model.forward_classifier(features)
                cil_outputs = logits[..., : (self.args.task + 1) * self.args.class_num]
                til_outputs = logits[..., eval_t * self.args.class_num: (eval_t+1) * self.args.class_num]
                _, cil_prediction = torch.max(torch.softmax(cil_outputs, dim=1), dim=1)
                _, til_prediction = torch.max(torch.softmax(til_outputs, dim=1), dim=1)
                til_prediction += eval_t * self.args.class_num
                
                references = batch[1]
                total_num += batch[0].shape[0]

                label_list += references.cpu().numpy().tolist()
                cil_prediction_list += cil_prediction.cpu().numpy().tolist()
                til_prediction_list += til_prediction.cpu().numpy().tolist()

        cil_accuracy = sum(
            [1 if label_list[i] == cil_prediction_list[i] else 0 for i in range(total_num)]
        ) / total_num

        til_accuracy = sum(
            [1 if label_list[i] == til_prediction_list[i] else 0 for i in range(total_num)]
        ) / total_num

        tp_accuracy = sum(
            [1 if cil_prediction_list[i] // self.args.class_num == eval_t else 0 for i in range(total_num)]
        ) / total_num
    
        results = {
            'til_accuracy': round(til_accuracy, 4),
            'cil_accuracy': round(cil_accuracy, 4),
            'TP_accuracy': round(tp_accuracy, 4)
        }
        return results