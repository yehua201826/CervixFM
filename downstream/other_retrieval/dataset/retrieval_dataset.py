from torch.utils.data import Dataset
import os
import numpy as np
import random
from torchvision import transforms
from PIL import Image


def data_label_efficiency(data_label, efficiency, seed=None, is_split=False):
    # if seed is None:
    #     np.random.seed(100 * efficiency)
    # else:
    #     np.random.seed(seed) # 42 52 62 72
    #     perm = np.random.permutation(len(data_label))
    #     data_label = data_label[perm]
    data_label = np.array(data_label)
    if seed is None:
        np.random.seed(int(100 * efficiency))
        if efficiency < 1 and is_split==False:
            perm = np.random.permutation(len(data_label))
            data_label = data_label[perm]

    else:
        if is_split:
            np.random.seed(seed)
        else:
            np.random.seed(int(seed + 100 * efficiency)) # 42 52 62 72

        perm = np.random.permutation(len(data_label))
        data_label = data_label[perm]

    new_data_label = []

    labels = data_label[:,1]
    for label in np.unique(labels):
        label_data = data_label[labels==label]
        label_sample_size = int(len(label_data) * efficiency)
        new_data_label.extend(label_data[:label_sample_size])

    # random.shuffle(new_data_label)
    return new_data_label


def split_train_test(data_label, seed):
    train_split = data_label_efficiency(data_label, 0.8, seed, is_split=True)

    data_label_tuples = [tuple(x) for x in data_label]
    train_split_tuples = [tuple(x) for x in train_split]
    test_split = list(set(data_label_tuples) - set(train_split_tuples))
    print('------------------train/test-----------------')
    print(len(train_split), len(test_split))
    return train_split, test_split


def extract_data_label(data_path, stage=None, train=False, test=False, is_HiCervix=False, seed=None):
    img_names = []
    img_labels = []
    if stage:
        npy_path = os.path.join(data_path, stage + '.npy')
        data_label = np.load(npy_path)

        if is_HiCervix:
            for d_l in data_label:
                [img_path, label_1, label_2, label_3] = d_l
                img_names.append(img_path)
                img_labels.append([int(label_1), int(label_2), int(label_3)])
        else:
            for d_l in data_label:
                [img_path, img_label] = d_l
                img_names.append(img_path)
                img_labels.append(int(img_label))

    else:
        npy_path = os.path.join(data_path, 'data_label.npy')
        data_label = np.load(npy_path)

        train_data_label, test_data_label = split_train_test(data_label, seed)

        if train:
            for d_l in train_data_label:
                [img_path, img_label] = d_l
                img_names.append(img_path)
                img_labels.append(int(img_label))

        if test:
            for d_l in test_data_label:
                [img_path, img_label] = d_l
                img_names.append(img_path)
                img_labels.append(int(img_label))

    return img_names, img_labels


class Retrieval_Dataset(Dataset):

    def __init__(self, data_path, stage=None, train=False, test=False, is_HiCervix=False, seed=None):

        if stage:
            assert stage in ['train', 'test'], '"stage" must be "train" or "test" !'
        else:
            assert sum([train, test]) == 1, 'One of "train" and "test" must be True !'

        self.data_path = os.path.join(data_path, stage or 'data')
        self.data, self.label = extract_data_label(data_path, stage, train=train, test=test, is_HiCervix=is_HiCervix, seed=seed)

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, item):

        img_name = self.data[item]
        img_path = os.path.join(self.data_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tran = self.transformer(img)
        img_label = self.label[item]
        return img_name, img_tran, img_label