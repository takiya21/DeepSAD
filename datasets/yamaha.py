# coding:utf-8
import os
import torch
import torchvision.transforms as transforms
import itertools
import numpy as np
from torchvision import datasets as datasets
from torch.utils import data
from sklearn.model_selection import StratifiedKFold
from PIL import Image

class Mydatasets(data.Dataset):
    def __init__(self, root, height, width, approach, is_train=True):
        self.dataset_path = root
        self.approach = approach
        self.is_train = is_train

        self.data, self.label = self.load_dataset_folder()

        self.transform = transforms.Compose([
            transforms.Resize((height, width), Image.ANTIALIAS),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        data = Image.open(data).convert('RGB')
        data = self.transform(data)

        return data, label, idx

    def __len__(self):
        return len(self.label)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        data, label = [], []

        img_dir = os.path.join(self.dataset_path, phase)
        # print('img_dir:',img_dir) # ../../data/No23/23_template_matching/fuse/train

        img_types = ['NG', 'OK']#sorted(os.listdir(img_dir))
        #print('img_types:',img_types) # ['0_normal_534', '1_outlier_14']
        for _, img_type in enumerate(img_types):
            img_type_dir = os.path.join(img_dir, img_type)
            # print('img_type_dir:',img_type_dir) 
            # ../../data/No23/23_template_matching/fuse/train/0_normal_534
            # ../../data/No23/23_template_matching/fuse/train/1_outlier_14
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)])
            #print(img_fpath_list)
            # ['../../data/No23/23_template_matching/fuse/train/0_normal_534/oo.png', ...,
            #  '../../data/No23/23_template_matching/fuse/train/0_normal_534/oo.png']
            data.extend(img_fpath_list)

            # load gt labels
            if img_type == img_types[0]:
                label.extend([0] * len(img_fpath_list))
            elif img_type == img_types[1]:
                # if "TrueDSVDD0" not in c: #���}���u
                
                label.extend([1] * len(img_fpath_list))
            #else:
            #    label.extend([2] * len(img_fpath_list))


        assert len(data) == len(label), 'number of x and y should be same'
        print("len(data),label:",len(data))

        return list(data), list(label)

class Mydatasets2(torch.utils.data.Dataset):

    def __init__(self, data, label, height, width):
        self.data = data
        self.label = label
        self.height = height
        self.width = width

        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width), Image.ANTIALIAS),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, idx):
        data, label = self.data[idx], self.label[idx]
        data = Image.open(data).convert('RGB')
        data = self.transform(data)

        return data, label, idx

    def __len__(self):
        return len(self.label)


def yamaha_dataset(root, approach: str, seed, height, width, normal_class: int, known_outlier_class: int):
    dataset = Mydatasets(root, height, width, approach, is_train=True)
    data = dataset.data
    label = dataset.label
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    k = 0
    for train_index, val_index in kf.split(data, label):
        train_data = []
        train_label = []
        val_data = []
        val_label = []
        for _, j in enumerate(train_index):
            if (approach=='DSVDD'):
                if (label[j]==0): # ����f�[�^�̂ݒ��o
                    train_data.append(data[j])
                    train_label.append(label[j])
            else:
                train_data.append(data[j])
                train_label.append(label[j])

        for _, j in enumerate(val_index):
                val_data.append(data[j])
                val_label.append(label[j])

        if(k==0):
            train_dataset0 = Mydatasets2(train_data, train_label, height, width)
            val_dataset0 = Mydatasets2(val_data, val_label, height, width)
        elif(k==1):
            train_dataset1 = Mydatasets2(train_data, train_label, height, width)
            val_dataset1 = Mydatasets2(val_data, val_label, height, width)
        else:
            train_dataset2 = Mydatasets2(train_data, train_label, height, width)
            val_dataset2 = Mydatasets2(val_data, val_label, height, width)
        k += 1 

    train_dataset = [] # train_dataset = [[train_dataset0], [train_dataset1], [train_dataset2]]
    val_dataset = []
    train_dataset.append(train_dataset0)
    train_dataset.append(train_dataset1)
    train_dataset.append(train_dataset2)

    val_dataset.append(val_dataset0)
    val_dataset.append(val_dataset1)
    val_dataset.append(val_dataset2)

    print(len(train_dataset[0]))
    print(len(train_dataset[1]))
    print(len(train_dataset[2]))
    print(len(val_dataset[0]))
    print(len(val_dataset[1]))
    print(len(val_dataset[2]))
    
    test_dataset = Mydatasets(root, height, width, approach, is_train=False)
    classes = ('0', '1')
    return train_dataset, val_dataset, test_dataset, classes

# def channel3_closs_b(root, approach: str, seed, height, width, normal_class: int, known_outlier_class: int):
#     dataset = Mydatasets(root, height, width, approach, is_train=True)
#     data = np.array(dataset.data)
#     label = np.array(dataset.label)

#     train_dataset = []
#     val_dataset = []
#     one_set_data = []
#     one_set_label = []
#     train_idx = []
#     val_idx = []

#     kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed) # No.6�ُ̈�f�[�^�̐�:7
#     for _, val_index in kf.split(data, label):
#         idx_list = []
#         for _, j in enumerate(val_index):
#             idx_list.append(j)
#         idx_list = np.array(idx_list)  # 7�������āA1�Z�b�g���擾
#         one_set_data.append(data[idx_list])
#         one_set_label.append(label[idx_list])

#     one_set_data = np.array(one_set_data)
#     one_set_label = np.array(one_set_label)

#     mylist = np.array(range(len(one_set_label))) # 4:3�ɕ����邽�߂̃C���f�b�N�X�擾�p
#     all = itertools.combinations(mylist, 4)
#     for x in all: # 4:3�̃C���f�b�N�X
#         x = np.array(x)
#         train_idx.append(x) # 4
#         val_idx.append(np.setdiff1d(mylist, x)) # 7-4

#     print(val_idx)
#     train_data = one_set_data[train_idx]
#     train_label = one_set_label[train_idx]
#     val_data = one_set_data[val_idx]
#     val_label = one_set_label[val_idx]
#     print(type(train_data))

#     for t_data, t_label, v_data, v_label in zip(train_data, train_label, val_data, val_label):
#         print(t_data.shape)
#         t_data = np.array(sum(t_data.tolist(), []))
#         train_dataset.append(Mydatasets2(t_data, t_label, height, width))
#         val_dataset.append(Mydatasets2(v_data, v_label, height, width))
    
#     test_dataset = Mydatasets(root, height, width, approach, is_train=False)

#     print(val_dataset[0])

#     classes = ('0', '1')

#     return train_dataset, val_dataset, test_dataset, classes