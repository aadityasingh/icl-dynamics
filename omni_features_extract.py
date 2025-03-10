'''
This file loads the Omniglot dataset and extracts Resnet18 features for each image.
It should be easily extensible to other pre-trained encoders.

As rotations and flips of Omniglot images were used for the transience paper,
this file extracts 1623 * 20 * 8 = 259,680 vectors in its full capacity.
'''

import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Subset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
import pickle as pkl
from torchvision.models import ResNet18_Weights, AlexNet_Weights, ResNet50_Weights, alexnet, resnet18, resnet50
from tqdm import tqdm
from omniglot_dataset import OmniglotFull, RotateAndFlipTransform, RotateAndFlipDataset
import pdb

base_path = '/path/to/save'
data_save_path = f'{base_path}/data'

data_transforms = dict()
for resize in [224]:
  data_transforms[resize] = transforms.Compose([
    transforms.Resize(resize),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

# Load Omniglot dataset
input_datasets = dict()
rotate_and_flip_transform = RotateAndFlipTransform()
for t in data_transforms:
  input_datasets[t] = RotateAndFlipDataset(
    OmniglotFull(
      root=data_save_path,
      transform=data_transforms[t]),
    transform=rotate_and_flip_transform)

models = dict()

rnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
if torch.cuda.is_available():
  rnet18 = rnet18.to('cuda')
a = rnet18.fc
rnet18.fc = nn.Identity()
rnet18.eval()
models['resnet18'] = [rnet18, a]

# For data loader
# For resnet18, up to batch size 2048 is feasible -- might need tuning based on specific hardware
batch_size = 2048
# Number of omniglot examples to grab
# NOTE: the file used for the IH paper only had 5 extracted exemplars (only 1 was used)
# NOTE: for the transience and coopetition paper, num_examples = 20 (the max) was used
num_examples = 20

results = dict()
print(len(input_datasets))
for model in models:
  results[model] = dict()
  for t in input_datasets:
    results[model][t] = dict()
    for i in range(num_examples):
      results[model][t][i] = dict()
      results[model][t][i]['feat'] = []
      results[model][t][i]['last'] = []
      
with torch.no_grad():
  for input_idx, t in enumerate(input_datasets):
    print("working on input dataset ", input_idx)
    for i in range(num_examples):
      print("working on example:", i)
      data_loader = torch.utils.data.DataLoader(
                Subset(input_datasets[t], 
                     np.arange(i, len(input_datasets[t]), 20)), 
                batch_size=batch_size)
      for b, l in tqdm(data_loader):
        if torch.cuda.is_available():
          b = b.to('cuda')
        for m in models:
          feat = models[m][0](b)
          # Get features
          results[m][t][i]['feat'].append(feat.to('cpu'))
          # Feed through output projection
          results[m][t][i]['last'].append(models[m][1](feat).to('cpu'))

f = h5py.File(f'{base_path}/embeddings/omniglot_features_all.h5', 'w')
for m in results:
  for t in results[m]:
    feat, last = [], []
    for i in results[m][t]:
      feat.append(np.concatenate(results[m][t][i]['feat']))
      last.append(np.concatenate(results[m][t][i]['last']))
    f.create_dataset('{}/{}/feat'.format(m,t), data = np.stack(feat, axis=1))
    f.create_dataset('{}/{}/last'.format(m,t), data = np.stack(last, axis=1))


train_class_inds = (np.arange(1600)[None,:] + (1623*np.arange(8))[:, None]).reshape(-1)
test_class_inds = (np.arange(1600, 1623)[None,:] + (1623*np.arange(8))[:, None]).reshape(-1)


reordered = h5py.File(f'{base_path}/embeddings/omniglot_features_reordered.h5', 'w')
def reorder(name, obj):
  if isinstance(obj, h5py.Dataset):
    reordered.create_dataset(name, data=np.concatenate([obj[train_class_inds], obj[test_class_inds]],axis=0))

f.visititems(reorder)


norotate = h5py.File(f'{base_path}/embeddings/omniglot_features_norotate.h5', 'w')
def reorder(name, obj):
  if isinstance(obj, h5py.Dataset):
    norotate.create_dataset(name, data=obj[:1623])
f.visititems(reorder)

f.close()
reordered.close()
norotate.close()