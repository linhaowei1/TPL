from itertools import chain
import os
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
from copy import deepcopy
from PIL import Image
import math
import torch
import PIL


def get_MNIST_transform(args):

    TRANSFORM = transforms.Compose(
        [
            transforms.Resize(size=248, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                                 std=torch.tensor([0.2290, 0.2240, 0.2250])),
        ]
    )
    return TRANSFORM, TRANSFORM


def get_transform(args):

    config = {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (
        0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 0.9}
    TRANSFORM = create_transform(**config)
    return TRANSFORM, TRANSFORM


def get_dataset(args):

    f_name = os.path.join('./sequence', args.sequence_file)

    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
    dataset_name = random_sep[0]

    if 'C10-5T' in dataset_name:
        DATA_PATH = '~/data'
        args.total_class = 10
        args.class_num = int(args.total_class / args.ntasks)
        args.mean = (0.4914, 0.4822, 0.4465)
        args.std = (0.2023, 0.1994, 0.2010)
        train_transform, test_transform = get_transform(args)
        train = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(DATA_PATH, train=False, download=False, transform=test_transform)
        label_map = {
            0: list(range(10)),
            1: [3, 9, 1, 8, 0, 2, 6, 4, 5, 7],
            2: [6, 0, 2, 8, 1, 9, 7, 3, 5, 4],
            3: [2, 6, 1, 5, 9, 8, 0, 4, 3, 7],
            4: [1, 5, 7, 2, 0, 3, 4, 6, 8, 9]
        }

        train.targets = [label_map[args.class_order].index(x) for x in train.targets]
        test.targets = [label_map[args.class_order].index(x) for x in test.targets]

    elif 'C100-' in dataset_name:
        DATA_PATH = '~/data'
        args.total_class = 100
        args.class_num = int(args.total_class / args.ntasks)
        args.mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        args.std = [x / 255 for x in [68.2, 65.4, 70.4]]
        train_transform, test_transform = get_transform(args)
        train = datasets.CIFAR100(DATA_PATH, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR100(DATA_PATH, train=False, download=False, transform=test_transform)
        label_map = {
            0: list(range(100)),
            1: [49, 98, 97, 53, 48, 62, 89, 23, 82, 13, 40, 35, 71, 59, 34, 95, 67, 11, 27, 7, 47, 85, 36, 70, 51, 32, 60, 16, 29, 84, 39, 8, 17, 42, 72, 18, 15, 55, 83, 10, 37, 99, 66, 22, 14, 57, 24, 38, 80, 0, 52, 88, 77, 3, 50, 6, 41, 44, 93, 9, 96, 81, 45, 58, 5, 64, 86, 2, 78, 68, 75, 56, 46, 91, 43, 20, 87, 1, 33, 28, 19, 61, 30, 74, 65, 79, 63, 25, 12, 21, 31, 90, 69, 73, 4, 94, 76, 92, 54, 26],
            2: [64, 79, 89, 9, 88, 3, 26, 94, 61, 62, 73, 69, 83, 8, 75, 23, 45, 92, 74, 1, 84, 71, 96, 52, 7, 95, 2, 5, 70, 28, 77, 60, 43, 22, 91, 78, 34, 80, 48, 51, 58, 37, 6, 25, 85, 97, 40, 27, 32, 98, 36, 21, 39, 31, 15, 49, 66, 72, 67, 24, 20, 93, 87, 54, 90, 76, 99, 30, 53, 29, 82, 57, 65, 4, 19, 11, 14, 41, 16, 86, 59, 68, 35, 55, 38, 17, 33, 50, 81, 63, 42, 10, 18, 56, 44, 13, 46, 0, 47, 12],
            3: [97, 1, 48, 88, 58, 46, 87, 18, 35, 71, 45, 6, 31, 69, 21, 96, 9, 44, 14, 68, 98, 27, 56, 38, 13, 63, 47, 57, 22, 64, 8, 73, 78, 94, 52, 4, 23, 28, 85, 2, 19, 10, 92, 7, 93, 76, 42, 34, 49, 80, 40, 37, 66, 83, 33, 99, 36, 12, 41, 39, 75, 25, 3, 95, 16, 0, 29, 53, 60, 11, 24, 82, 86, 32, 91, 43, 65, 89, 15, 81, 17, 62, 90, 54, 51, 20, 55, 30, 77, 59, 50, 5, 74, 84, 67, 79, 70, 61, 72, 26],
            4: [34, 31, 97, 47, 83, 59, 39, 4, 32, 44, 26, 73, 45, 33, 56, 87, 82, 23, 88, 10, 51, 57, 65, 84, 43, 37, 9, 74, 28, 24, 90, 25, 60, 80, 5, 64, 63, 62, 40, 19, 49, 21, 77, 95, 99, 16, 12, 14, 70, 54, 53, 38, 8, 72, 18, 68, 15, 94, 36, 7, 1, 69, 2, 61, 98, 75, 85, 11, 17, 76, 22, 27, 92, 71, 3, 0, 66, 42, 96, 67, 35, 30, 46, 81, 48, 93, 79, 6, 13, 86, 20, 91, 78, 50, 89, 41, 52, 55, 29, 58]
        }

        train.targets = [label_map[args.class_order].index(x) for x in train.targets]
        test.targets = [label_map[args.class_order].index(x) for x in test.targets]

    elif dataset_name.startswith('T-'):
        args.total_class = 200
        args.class_num = int(args.total_class / args.ntasks)
        args.mean = (0.4914, 0.4822, 0.4465)
        args.std = (0.2023, 0.1994, 0.2010)
        train_transform, test_transform = get_transform(args)
        train = datasets.ImageFolder(root='~/data/tiny-imagenet-200/train', transform=train_transform)
        test = datasets.ImageFolder(root='~/data/tiny-imagenet-200/val', transform=test_transform)
        label_map = {
            0: list(range(200)),
            1: [117, 8, 183, 39, 40, 47, 75, 133, 193, 28, 130, 31, 98, 119, 188, 161, 57, 92, 54, 134, 6, 71, 147, 70, 139, 68, 77, 149, 17, 87, 132, 184, 59, 52, 194, 187, 159, 196, 166, 50, 63, 62, 141, 20, 126, 99, 19, 182, 164, 34, 2, 13, 97, 78, 151, 85, 150, 74, 111, 11, 61, 83, 41, 24, 55, 101, 110, 88, 60, 14, 65, 4, 51, 5, 30, 171, 158, 84, 15, 10, 46, 165, 118, 140, 90, 186, 107, 148, 180, 42, 152, 64, 189, 109, 136, 106, 91, 66, 178, 73, 172, 29, 25, 103, 44, 108, 191, 36, 72, 76, 82, 167, 160, 199, 9, 155, 175, 174, 179, 144, 177, 197, 170, 81, 121, 113, 58, 21, 89, 0, 69, 157, 137, 1, 26, 37, 153, 124, 143, 95, 23, 105, 79, 48, 32, 3, 190, 38, 135, 80, 198, 33, 53, 56, 49, 112, 125, 156, 131, 116, 129, 67, 162, 173, 123, 12, 181, 7, 192, 169, 185, 104, 100, 138, 168, 195, 43, 93, 45, 35, 22, 142, 146, 16, 127, 86, 128, 114, 27, 120, 145, 163, 102, 122, 18, 154, 94, 96, 115, 176],
            2: [121, 66, 149, 189, 103, 195, 0, 72, 179, 46, 7, 159, 70, 65, 123, 76, 54, 37, 186, 62, 96, 136, 124, 69, 181, 6, 57, 125, 161, 81, 134, 147, 132, 59, 20, 50, 93, 71, 117, 33, 135, 47, 36, 120, 94, 73, 75, 14, 102, 60, 113, 142, 175, 115, 184, 185, 152, 63, 12, 198, 24, 26, 119, 109, 165, 87, 144, 64, 48, 52, 21, 95, 116, 187, 137, 10, 84, 162, 90, 131, 88, 150, 110, 41, 146, 106, 86, 127, 151, 16, 107, 67, 129, 140, 4, 172, 39, 23, 51, 183, 197, 31, 157, 188, 171, 58, 13, 153, 32, 98, 173, 130, 97, 80, 133, 163, 53, 44, 141, 145, 155, 176, 156, 138, 22, 68, 112, 3, 174, 2, 42, 25, 29, 104, 170, 178, 193, 126, 122, 30, 196, 199, 182, 128, 91, 56, 49, 111, 83, 78, 89, 61, 192, 34, 148, 191, 180, 190, 74, 167, 158, 139, 1, 101, 166, 143, 28, 8, 43, 105, 38, 177, 118, 55, 108, 19, 5, 168, 15, 79, 160, 45, 169, 164, 85, 82, 77, 27, 40, 99, 92, 194, 18, 11, 154, 35, 100, 17, 9, 114],
            3: [156, 137, 7, 123, 154, 38, 121, 40, 43, 6, 76, 129, 91, 18, 12, 149, 162, 189, 145, 107, 5, 85, 78, 111, 191, 71, 146, 87, 155, 92, 48, 49, 21, 34, 23, 187, 179, 110, 102, 186, 105, 184, 29, 90, 159, 79, 28, 108, 89, 128, 57, 96, 194, 54, 55, 167, 141, 51, 67, 0, 177, 99, 26, 173, 1, 163, 122, 115, 30, 101, 170, 198, 134, 69, 61, 58, 192, 171, 185, 37, 124, 15, 114, 132, 181, 9, 157, 83, 19, 131, 73, 86, 153, 138, 32, 8, 33, 165, 42, 180, 44, 168, 188, 81, 64, 166, 24, 172, 142, 95, 35, 161, 160, 13, 119, 199, 39, 100, 97, 125, 52, 195, 65, 158, 197, 127, 46, 4, 175, 20, 56, 190, 41, 174, 151, 84, 182, 183, 109, 75, 3, 93, 106, 136, 50, 17, 74, 10, 150, 60, 112, 164, 193, 53, 14, 169, 152, 82, 116, 80, 63, 77, 120, 117, 11, 72, 31, 104, 113, 68, 144, 88, 178, 47, 16, 27, 176, 98, 148, 94, 25, 126, 143, 62, 118, 70, 140, 45, 66, 130, 196, 147, 133, 59, 103, 36, 139, 2, 135, 22],
            4: [187, 110, 38, 174, 97, 189, 39, 109, 122, 37, 42, 65, 101, 188, 134, 191, 153, 194, 3, 147, 78, 129, 52, 1, 185, 85, 22, 60, 98, 51, 155, 145, 24, 103, 2, 73, 139, 74, 18, 175, 48, 105, 46, 31, 161, 171, 14, 117, 69, 167, 12, 163, 25, 121, 13, 177, 16, 102, 56, 142, 107, 151, 53, 44, 62, 169, 176, 150, 67, 86, 91, 82, 5, 156, 128, 70, 149, 179, 144, 19, 146, 160, 21, 49, 0, 35, 119, 6, 141, 131, 94, 30, 162, 159, 76, 45, 17, 100, 118, 84, 66, 158, 15, 64, 54, 27, 89, 123, 193, 4, 80, 96, 58, 152, 93, 168, 108, 59, 113, 29, 34, 182, 83, 55, 11, 10, 111, 136, 133, 28, 192, 79, 127, 180, 140, 95, 68, 106, 61, 41, 157, 195, 90, 183, 130, 7, 125, 124, 40, 63, 116, 186, 199, 148, 120, 104, 75, 138, 178, 43, 181, 8, 143, 137, 20, 33, 99, 170, 184, 32, 87, 92, 154, 166, 88, 198, 26, 115, 190, 71, 72, 77, 50, 132, 165, 135, 164, 9, 47, 126, 57, 112, 114, 172, 197, 81, 36, 173, 23, 196],
        }

        train.samples = [(x[0], label_map[args.class_order].index(x[1])) for x in train.samples]
        test.samples = [(x[0], label_map[args.class_order].index(x[1])) for x in test.samples]
        train.targets = [label_map[args.class_order].index(x) for x in train.targets]
        test.targets = [label_map[args.class_order].index(x) for x in test.targets]

    else:
        raise NotImplementedError

    data = {}
    args.task2cls = [int(random_sep[t].split('-')[-1]) for t in range(args.ntasks)]
    cls_id_past = []
    for t in range(args.ntasks):
        data[t] = {}

        cls_id = [int(random_sep[t].split('-')[-1]) * args.class_num + i for i in range(args.class_num)]
        ## train
        train_ = deepcopy(train)

        targets_aux, data_aux, full_target_aux, names_aux = [], [], [], []
        idx_aux = []

        for c in cls_id:
            idx = np.where(np.array(train.targets) == c)[0]

            if dataset_name.startswith('T-'):   # for tinyImagenet
                idx_aux.append(idx)
            else:
                data_aux.append(train.data[idx])
                targets_aux.append(np.zeros(len(idx), dtype=np.int64) + c)
                full_target_aux.append([[c, c]] for _ in range(len(idx)))
                names_aux.append([str(c) for _ in range(len(idx))])

        if dataset_name.startswith('T-'):
            idx_list = np.concatenate(idx_aux)
            train_ = Subset(train_, idx_list)
            train_.data = []
            train_.targets = np.array(train_.dataset.targets)[idx_list]
            train_.transform = train_.dataset.transform
        elif dataset_name.startswith('M-'):
            train_.data = torch.from_numpy(np.concatenate(data_aux, 0))
            train_.targets = torch.from_numpy(np.concatenate(targets_aux, 0))
        else:
            train_.data = np.array(list(chain(*data_aux)))
            train_.targets = np.array(list(chain(*targets_aux)))
            train_.full_labels = np.array(list(chain(*full_target_aux)))
            train_.names = list(chain(*names_aux))
        del data_aux, targets_aux, full_target_aux, names_aux, idx_aux
        data[t]['train'] = train_

        ## test
        test_ = deepcopy(test)
        targets_aux, data_aux, full_target_aux, names_aux = [], [], [], []
        idx_aux = []
        for c in cls_id:
            idx = np.where(np.array(test.targets) == c)[0]
            if dataset_name.startswith('T-'):
                idx_aux.append(idx)
            else:
                data_aux.append(test.data[idx])
                targets_aux.append(np.zeros(len(idx), dtype=np.int64) + c)
                full_target_aux.append([[c, c]] for _ in range(len(idx)))
                names_aux.append([str(c) for _ in range(len(idx))])

        if dataset_name.startswith('T-'):
            idx_list = np.concatenate(idx_aux)
            test_ = Subset(test_, idx_list)
            test_.data = []
            test_.targets = np.array(test_.dataset.targets)[idx_list]
            test_.transform = test_.dataset.transform
        elif dataset_name.startswith('M-'):
            test_.data = torch.from_numpy(np.concatenate(data_aux, 0))
            test_.targets = torch.from_numpy(np.concatenate(targets_aux, 0))
        else:
            test_.data = np.array(list(chain(*data_aux)))
            test_.targets = np.array(list(chain(*targets_aux)))
            test_.full_labels = np.array(list(chain(*full_target_aux)))
            test_.names = list(chain(*names_aux))
        del data_aux, targets_aux, full_target_aux, names_aux
        data[t]['test'] = test_

        ## replay
        replay_ = deepcopy(train)
        targets_aux, data_aux, full_target_aux, names_aux = [], [], [], []
        idx_aux = []
        # mix replay dataset
        if t > 0 and 'deit' in args.baseline:
            for c in cls_id_past:
                idx = np.where(np.array(train.targets) == c)[0][:(args.replay_buffer_size // len(cls_id_past))]
                if dataset_name.startswith('T-'):
                    idx_aux.append(idx)
                else:
                    data_aux.append(train.data[idx])
                    targets_aux.append(np.zeros(len(idx), dtype=np.int64) + c)
                    full_target_aux.append([[c, c]] for _ in range(len(idx)))
                    names_aux.append([str(c) for _ in range(len(idx))])

            if dataset_name.startswith('T-'):
                idx_list = np.concatenate(idx_aux)
                replay_ = Subset(replay_, idx_list)
                replay_.data = []
                replay_.targets = np.array(replay_.dataset.targets)[idx_list]
                replay_.transform = replay_.dataset.transform
            elif dataset_name.startswith('M-'):
                replay_.data = torch.from_numpy(np.concatenate(data_aux, 0))
                replay_.targets = torch.from_numpy(np.concatenate(targets_aux, 0))
            else:
                replay_.data = np.array(list(chain(*data_aux)))
                replay_.targets = np.array(list(chain(*targets_aux)))
                replay_.full_labels = np.array(list(chain(*full_target_aux)))
                replay_.names = list(chain(*names_aux))
        else:
            replay_ = None
        del data_aux, targets_aux, full_target_aux, names_aux
        data[t]['replay'] = replay_

        cls_id_past += [int(random_sep[t].split('-')[-1]) * args.class_num + i for i in range(args.class_num)]

    data[args.ntasks] = {}
    ## replay for the final task
    replay_ = deepcopy(train)
    targets_aux, data_aux, full_target_aux, names_aux = [], [], [], []
    idx_aux = []
    # mix replay dataset
    for c in cls_id_past:
        idx = np.where(np.array(train.targets) == c)[0][:(args.replay_buffer_size // len(cls_id_past))]
        if dataset_name.startswith('T-'):
            idx_aux.append(idx)
        else:
            data_aux.append(train.data[idx])
            targets_aux.append(np.zeros(len(idx), dtype=np.int64) + c)
            full_target_aux.append([[c, c]] for _ in range(len(idx)))
            names_aux.append([str(c) for _ in range(len(idx))])

    if dataset_name.startswith('T-'):
        idx_list = np.concatenate(idx_aux)
        replay_ = Subset(replay_, idx_list)
        replay_.data = []
        replay_.targets = np.array(replay_.dataset.targets)[idx_list]
        replay_.transform = replay_.dataset.transform
    elif dataset_name.startswith('M-'):
        replay_.data = torch.from_numpy(np.concatenate(data_aux, 0))
        replay_.targets = torch.from_numpy(np.concatenate(targets_aux, 0))
    else:
        replay_.data = np.array(list(chain(*data_aux)))
        replay_.targets = np.array(list(chain(*targets_aux)))
        replay_.full_labels = np.array(list(chain(*full_target_aux)))
        replay_.names = list(chain(*names_aux))

    del data_aux, targets_aux, full_target_aux, names_aux
    data[args.ntasks]['replay'] = replay_

    return data
