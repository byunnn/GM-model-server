# 이 파이썬 코드는 PyTorch의 데이터셋을 구성하기 위한 클래스를 정의하고 있습니다. 주요 기능은 이미지 파일들을 불러오고, 이를 transform으로 전처리하는 작업을 수행합니다.

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset): # PyTorch의 Dataset 클래스를 상속하여 새로운 데이터셋 클래스를 정의합니다.
    def __init__(self, root, transform=None):
        self.samples = listdir(root) # 지정된 root 디렉토리에서 이미지 파일을 찾습니다.
        self.samples.sort()  # 이미지 파일 이름을 정렬합니다.
        self.transform = transform  # 이미지에 적용할 변환(transform)을 저장합니다.
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]  # index에 해당하는 이미지 파일의 이름을 가져옵니다.
        img = Image.open(fname).convert('RGB') # 이미지를 RGB 형태로 불러옵니다.
        if self.transform is not None:
            img = self.transform(img) # 지정된 transform이 있다면 이를 이미지에 적용합니다.
        return img

    def __len__(self):
        return len(self.samples)  # 데이터셋에 포함된 샘플의 수를 반환합니다.


class ReferenceDataset(data.Dataset): # PyTorch의 Dataset 클래스를 상속하여 새로운 데이터셋 클래스를 정의합니다.
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root) # 지정된 root 디렉토리에서 데이터셋을 만듭니다.
        self.transform = transform # 이미지에 적용할 변환(transform)을 저장합니다.

    def _make_dataset(self, root): # root 디렉토리 내의 하위 디렉토리 이름(도메인)을 가져옵니다.
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], [] # 파일 이름과 레이블을 저장할 리스트를 초기화합니다.
        for idx, domain in enumerate(sorted(domains)): # 각 도메인에 대해 반복합니다.
            class_dir = os.path.join(root, domain) # 각 도메인 디렉토리의 경로를 만듭니다.
            cls_fnames = listdir(class_dir)  # 각 도메인 디렉토리에서 이미지 파일을 찾습니다.
            fnames += cls_fnames # 파일 이름을 fnames 리스트에 추가합니다.
            fnames2 += random.sample(cls_fnames, len(cls_fnames)) # 무작위로 선택된 파일 이름을 fnames2 리스트에 추가합니다.
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)

    transform = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
