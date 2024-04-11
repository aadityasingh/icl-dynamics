'''
A file for loading Omniglot, as well as creating a version that contains
8 version of each image (rotations + flips).
'''
import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class RotateAndFlipTransform:
  def __call__(self, img):
    rotations = [0, 90, 180, 270]
    flipped = [False, True]

    transformed_imgs = []

    for rotation in rotations:
      for flip in flipped:
        transformed_img = transforms.functional.rotate(img, rotation)
        if flip:
          transformed_img = transforms.functional.hflip(transformed_img)
        transformed_imgs.append(transformed_img)

    return transformed_imgs
  
class RotateAndFlipDataset(torch.utils.data.Dataset):
  def __init__(self, original_dataset, transform=None):
    self.original_dataset = original_dataset
    self.transform = transform

  def __getitem__(self, index):
    original_index = index // 8
    img, label = self.original_dataset[original_index]

    if self.transform:
      imgs = self.transform(img)
      img = imgs[index % 8]

    return img, label

  def __len__(self):
    return len(self.original_dataset) * 8


class OmniglotFull(torch.utils.data.Dataset):
  def __init__(self, root='./data', transform=transforms.ToTensor()):
    self.dataset_bkg = datasets.Omniglot(root=root, transform=transform, download=True)
    self.dataset_eval = datasets.Omniglot(root=root, transform=transform, background=False, download=True)
    self.bkg_classes = self.dataset_bkg[-1][1] + 1
    self.total_classes = self.bkg_classes + self.dataset_eval[-1][1] + 1
    
  def __getitem__(self, index):
    if index < len(self.dataset_bkg):
      image, label = self.dataset_bkg[index]
      return image, label
    else:
      image, label = self.dataset_eval[index - len(self.dataset_bkg)]
      return image, label + self.bkg_classes

  def __len__(self):
    return len(self.dataset_bkg) + len(self.dataset_eval)

if __name__ == '__main__':
  raw_dataset = OmniglotFull(transform=transforms.ToTensor())

  dim = raw_dataset[0][0].shape[1]
  plt.imshow(np.broadcast_to(np.transpose(raw_dataset[0][0], (1,2,0)), (dim,dim,3)))
  all_labels = np.array([x[1] for x in raw_dataset])
  all_images = np.concatenate([x[0] for x in raw_dataset])

  normalize_image_batch = (lambda x: x/np.sqrt(np.sum(x**2, axis=(1,2)))[:,None,None])
  one_per_class = all_images[::20]
  normalized_one_per_class = normalize_image_batch(one_per_class)
  one_per_class2 = all_images[1::20]
  normalized_one_per_class2 = normalize_image_batch(one_per_class2)

  u,s,vh = np.linalg.svd(one_per_class.reshape(1623, -1))
  un,sn,vhn = np.linalg.svd(normalized_one_per_class.reshape(1623,-1))
  u2,s2,vh2 = np.linalg.svd(one_per_class2.reshape(1623, -1))
  un2,sn2,vhn2 = np.linalg.svd(normalized_one_per_class2.reshape(1623,-1))

  print(s[:20], sn[:20], s2[:20], sn2[:20])

  to_plot=5
  fig,ax = plt.subplots(4, 5)
  for i in range(to_plot):
    ax[0,i].imshow(vh[i,:].reshape(dim,dim), cmap='gray')
    ax[1,i].imshow(vhn[i,:].reshape(dim,dim), cmap='gray')
    ax[2,i].imshow(vh2[i,:].reshape(dim,dim), cmap='gray')
    ax[3,i].imshow(vhn2[i,:].reshape(dim,dim), cmap='gray')

  plt.show()
