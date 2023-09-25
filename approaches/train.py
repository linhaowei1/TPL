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
        scheduler = None

        optimizer = SGD(model.adapter_parameters(), lr=self.args.learning_rate,
                        momentum=0.9, weight_decay=5e-4, nesterov=True)
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        model = model.cuda()
        if replay_loader is not None:
            replay_iterator = iter(replay_loader)
        before_train.prepare(self.args, model)

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

                s = (self.args.smax - 1 / self.args.smax) * step / len(
                    train_loader) + 1 / self.args.smax
                
                if replay_loader is not None:
                    try:
                        replay_batch = next(replay_iterator)
                        batch[0] = torch.cat((batch[0], replay_batch[0]), dim=0)
                        batch[1] = torch.cat((batch[1], replay_batch[1]), dim=0)
                    except:
                        replay_iterator = iter(replay_loader)
                        replay_batch = next(replay_iterator)
                        batch[0] = torch.cat((batch[0], replay_batch[0]), dim=0)
                        batch[1] = torch.cat((batch[1], replay_batch[1]), dim=0)

                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                features, masks = model.forward_features(self.args.task, batch[0], s=s)
                outputs = model.forward_classifier(self.args.task, features)
                loss = nn.functional.cross_entropy(outputs, batch[1])

                loss += HAT_reg(self.args, masks)

                loss.backward()

                if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    compensation(model, self.args, thres_cosh=self.args.thres_cosh, s=s)
                    optimizer.step(hat=(self.args.task > 0))
                    compensation_clamp(model, thres_emb=6)

                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while
            
            if scheduler is not None:
                scheduler.step()
            if completed_steps >= self.args.max_train_steps:
                break

        after_train.compute(self.args, model)

        for eval_t in range(self.args.task + 1):
            results = self.eval_cl(model, test_loaders, eval_t)
            print("*task {}, til_acc = {}, cil_acc = {}, tp_acc = {}".format(
                eval_t, results['til_accuracy'], results['cil_accuracy'], results['TP_accuracy']))
            utils.write_result(results, eval_t, self.args)

    def eval_cl(self, model, test_loaders, eval_t):

        model.eval()
        dataloader = test_loaders[eval_t]
        label_list = []
        prediction_list = []
        taskscore_list = []
        total_num = 0
        for task_mask in range(self.args.task + 1):
            total_num = 0
            task_pred = []
            task_confidence = []
            task_label = []
            for _, batch in enumerate(dataloader):
                with torch.no_grad():

                    features, _ = model.forward_features(task_mask, batch[0].cuda(), s=self.args.smax)
                    outputs = model.forward_classifier(task_mask, features)[
                        :, task_mask * self.args.class_num: (task_mask+1) * self.args.class_num]
                    score, prediction = torch.max(torch.softmax(outputs, dim=1), dim=1)

                    predictions = prediction + task_mask * self.args.class_num
                    references = batch[1]

                    total_num += batch[0].shape[0]
                    task_confidence += score.cpu().numpy().tolist()
                    task_label += references.cpu().numpy().tolist()
                    task_pred += predictions.cpu().numpy().tolist()

            label_list = task_label
            prediction_list.append(task_pred)
            taskscore_list.append(np.array(task_confidence))

        task_pred = np.argmax(np.stack(taskscore_list, axis=0), axis=0)
        cil_pred = [prediction_list[task_pred[i]][i] for i in range(total_num)]
        til_pred = [prediction_list[eval_t][i] for i in range(total_num)]

        cil_accuracy = sum(
            [1 if label_list[i] == cil_pred[i] else 0 for i in range(total_num)]
        ) / total_num
        til_accuracy = sum(
            [1 if label_list[i] == til_pred[i] else 0 for i in range(total_num)]
        ) / total_num
        TP_accuracy = sum(
            [1 if task_pred[i] == eval_t else 0 for i in range(total_num)]
        ) / total_num

        results = {
            'til_accuracy': round(til_accuracy, 4),
            'cil_accuracy': round(cil_accuracy, 4),
            'TP_accuracy': round(TP_accuracy, 4)
        }
        return results
