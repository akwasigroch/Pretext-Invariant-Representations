import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.models.resnet import resnet50

from utils import (AverageMeter, Logger, Memory, ModelCheckpoint,
                   NoiseContrastiveEstimator, Progbar, pil_loader)

device = torch.device('cuda:2')
data_dir = '/media/dysk/datasets/isic_challenge_2017/train'
negative_nb = 1000  # number of negative examples in NCE
lr = 0.001
checkpoint_dir = 'rotation_models'
log_filename = 'pretraining_log_rotation'


class RotationLoader(DatasetFolder):
    def __init__(self, root_dir):
        super(RotationLoader, self).__init__(root_dir, pil_loader, extensions=('jpg'))
        self.root_dir = root_dir
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        original = self.loader(path)
        image = torchvision.transforms.Resize((300, 300))(original)
        image = torchvision.transforms.RandomCrop((224, 224))(image)

        rotation = torchvision.transforms.Resize((224, 224))(image)
        # augmentation - collor jitter
        image = self.color_transform(image)
        rotation = self.color_transform(rotation)
        # augmentation - flips
        image = self.flips[0](image)
        image = self.flips[1](image)
        # augmentation - rotation
        angles = [90, 180, 270]
        angle = random.choice(angles)
        rotation = torchvision.transforms.functional.rotate(rotation, angle)

        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        rotation = torchvision.transforms.functional.to_tensor(rotation)
        # normalize
        image = self.normalize(image)
        rotation = self.normalize(rotation)

        return {'original': image, 'rotation': rotation, 'index': index}


dataset = RotationLoader(data_dir)
train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, num_workers=32)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = resnet50()
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.projection_original_features = nn.Linear(2048, 128)

    def forward_once(self, x):
        return self.network(x)

    def return_reduced_image_features(self, original):
        features = self.forward_once(original)
        features = features.view(-1, 2048)
        features = self.projection_original_features(features)
        return features

    def forward(self, images=None, rotation=None, mode=0):
        '''
        mode 0: get 128d feature for image,
        mode 1: get 128d feature for image and rotation

        '''
        if mode == 0:
            return self.return_reduced_image_features(images)
        if mode == 1:
            image_features = self.return_reduced_image_features(images)
            rotation_features = self.return_reduced_image_features(rotation)
            return image_features, rotation_features


net = Network().to(device)

optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

memory = Memory(size=len(dataset), weight=0.5, device=device)
memory.initialize(net, train_loader)


checkpoint = ModelCheckpoint(mode='min', directory=checkpoint_dir)
noise_contrastive_estimator = NoiseContrastiveEstimator(device)
logger = Logger(log_filename)

loss_weight = 0.5

for epoch in range(1000):
    print('\nEpoch: {}'.format(epoch))
    memory.update_weighted_count()
    train_loss = AverageMeter('train_loss')
    bar = Progbar(len(train_loader), stateful_metrics=['train_loss', 'valid_loss'])

    for step, batch in enumerate(train_loader):

        # prepare batch
        images = batch['original'].to(device)
        rotation = batch['rotation'].to(device)
        index = batch['index']
        representations = memory.return_representations(index).to(device).detach()
        # zero grad
        optimizer.zero_grad()

        #forward, loss, backward, step
        output = net(images=images, rotation=rotation, mode=1)

        loss_1 = noise_contrastive_estimator(representations, output[1], index, memory, negative_nb=negative_nb)
        loss_2 = noise_contrastive_estimator(representations, output[0], index, memory, negative_nb=negative_nb)
        loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2

        loss.backward()
        optimizer.step()

        # update representation memory
        memory.update(index, output[0].detach().cpu().numpy())

        # update metric and bar
        train_loss.update(loss.item(), images.shape[0])
        bar.update(step, values=[('train_loss', train_loss.return_avg())])
    logger.update(epoch, train_loss.return_avg())

    # save model if improved
    checkpoint.save_model(net, train_loss.return_avg(), epoch)
