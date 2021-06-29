import os
import random

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from matplotlib import pyplot as plt
import openslide

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import misc


def _get_slide_id(x):
  return '_'.join(os.path.basename(x).split('-')[0].split('_')[:5])

def _get_label_from_filepath(x):
  if os.path.isabs(x):
    x = os.path.basename(x)
  if x.startswith('MM_sarc'):
    return 'tumor'
  elif x.startswith(('BM_spin', 'N_lung')):
    return 'benign'
  else:
    print('unable to verify label for file:', x, flush=True)
    raise 


class CustomRotation:
    """Randomly rotate by right angles only."""
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class SlideDataset(Dataset):
  """ Extension of Pytorch dataset
  
  """
  def __init__(self, img_list, img_to_label, label_to_value, 
               batch_size=32, img_size=512, 
               n_samples=None, transforms=None, 
               process_full_img_list=True ):
    """
    # Args
      tile_dir
      slide_list
      img_to_label: dictionary matching each slide to label
      label_to_value: dictionary to convert text label to numerical value
      img_size
      batch_size
      n_samples: number of batches per epoch; only relevant if 
          process_full_img_list is False
      transforms: all transforms to apply to image must be passed here
      process_full_img_list: overrid n_samples and train on the entire 
          image list
    """
    self.img_list = img_list
    self.img_to_label = img_to_label
    self.label_to_value = label_to_value
    self.img_size = img_size
    self.n_samples = n_samples
    self.transforms = transforms
    self.process_full_img_list = process_full_img_list
    self.batch_size = batch_size
    self._debug_mode = False

  def __len__(self):
    # total number of images processed per epoch
    if self.process_full_img_list:
      return len(self.img_list)
    else:
      return self.n_samples

  def __getitem__(self, idx): 
    if self.process_full_img_list:
      img_path = self.img_list[idx]
    else:
      img_path = random.choice(self.img_list)
    img = Image.open(img_path)
    if self.transforms:
      img = self.transforms(img)
    img = img.reshape((3, self.img_size, self.img_size))
    #slide_id = _get_slide_id(img_path)
    label = _get_label_from_filepath(img_path)
    #label = self.img_to_label[os.path.basename(img_path)]
    label = torch.LongTensor(np.array(self.label_to_value[label]))
    #assert _get_label_from_filename(os.path.basename(img_path)) == label
    if self._debug_mode:
      print(img.shape, label.shape, label)
    return img, label
  
  def _verify_channel_first(self, x):
    channel_axis = np.argmin(x.shape)
    if channel_axis != 0:
      x = np.rollaxis(x, channel_axis, 0)
    return x

  def _verify_channel_last(self, x):
    channel_axis = np.argmin(x.shape)
    if channel_axis != 2:
      x = np.rollaxis(x, channel_axis, 3)
    return x

def _img_path_to_case(x):
  return '_'.join(os.path.basename(x).split('_')[:4])


def create_dataloader(phase, img_list, img_to_label, label_to_value,
                      img_size=512, batch_size=16, n_workers=6, 
                      n_samples=None, color_jitter=False, 
                      profiler_mode=False, max_imgs=None, **kwargs):
  """
  # Args
    phase: one of train, val, or test
    img_list
    img_to_label: dictionary mapping each image to its label
    label_to_value: dictionary mapping each label to its value
    img_size
    batch_size
    n_workers
    n_samples: total number of images to include
    color_jitter: apply color jitter data augmentation using 
        prespecified parameters
    profiler_mode
    max_imgs: maximum images PER CASE

  # Returns: a dataloader
  """

  if max_imgs:
    #TODO select subset of full image list
    cases = list(set([_img_path_to_case(x) for x in img_list]))
    list_ = []
    for case in cases:
      imgs = [x for x in img_list if case + '_' in x]
      random.shuffle(imgs)
      list_.extend(imgs[:max_imgs])
    img_list = list_

  normalization_transforms = [              
              # NOTE: transforms.ToTensor() Converts a PIL Image 
              # or numpy.ndarray  (H x W x C) in the range [0, 255] to 
              # a torch.FloatTensor of shape  (C x H x W) in the range 
              # [0.0, 1.0] if the PIL Image belongs to one of the several 
              # modes or if the numpy.ndarray has dtype = np.uint8
              # In the other cases, tensors are returned without scaling.
              transforms.ToTensor(),
              
              # required normalization values for use by torchvision 
              # pretrained models
              transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])]
  
  # color jitter parameters from Skrede et al, Lancet, 2020
  color_jitter_params = {'brightness': 0.1, 
                         'contrast':0, 
                         'saturation': 0.1, 
                         'hue': 0.05, }
  # make list of train transformations
  train_transforms = [CustomRotation(),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),]
  if color_jitter:
    train_transforms.append(transforms.ColorJitter(**color_jitter_params))
  train_transforms.extend(normalization_transforms)
  data_transforms = {
      'train':  transforms.Compose(train_transforms),
      'val':    transforms.Compose(normalization_transforms),
      'test':    transforms.Compose(normalization_transforms),
       }

  if not n_samples:
    n_samples = {'train': 1000, 'val': 1000, 'test': 1000}

  dataset = SlideDataset(img_list=img_list, 
                 img_to_label=img_to_label, 
                 label_to_value=label_to_value,
                 batch_size=batch_size,
                 img_size=img_size,
                 transforms=data_transforms[phase], 
                 n_samples=n_samples[phase], 
                 process_full_img_list = (False if profiler_mode else True),
                 )

  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           sampler=torch.utils.data.RandomSampler(dataset),
                           pin_memory=True, num_workers=n_workers)

  return dataloader

# —————————————————————————————————————————————————————————————————————————————
# Functions for viewing images generated by dataloader

def view_from_dataloader(dataloader):
  def show_landmarks_batch(img_batch, labels):
    """Show image with labels for a batch of samples."""
    batch_size = len(img_batch)
    im_size = img_batch.size(2)
    grid = utils.make_grid(img_batch, nrow=8)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

  for i, batch in enumerate(dataloader):
    print('\nbeginning batch {}'.format(i))
    imgs = batch[0]
    labels = batch[1]
    print(imgs.min(), imgs.max())
    # observe 4th batch and stop.
    plt.figure()
    show_landmarks_batch(imgs, labels)
    plt.axis('off')
    plt.show()
    if i == 3:
      break


def display_sample_images(dataset, batch_size=16, num_batches=3):
  """ Iterates through a dataset and displays images
  """
  fig = plt.figure(figsize=(8,8))
  for i in range(batch_size):
    print(i)
    img, label = dataset[i]
    # need to set channels last for plt.imshow
    #sample = _verify_channel_last(sample)
    img = np.rollaxis(img.numpy(), 0, 3)
    ax = fig.add_subplot(4, batch_size//4, i + 1)
    plt.tight_layout()
    #ax.set_title(labels[j])
    ax.axis('off')
    plt.imshow(img)
  #plt.show()


def save_sample_imgs(save_folder, dataset, batch_size=8, num_batches=3, 
                     display_only=False):
  """ Iterates through a dataset and displays images
  """
  fig = plt.figure(figsize=(8,8))
  for i in range(batch_size):
    print(i)
    img, label = dataset[i]
    # need to set channels last for plt.imshow
    #sample = _verify_channel_last(sample)
    img = np.rollaxis(img.numpy(), 0, 3)
    ax = fig.add_subplot(4, batch_size//4, i + 1)
    plt.tight_layout()
    ax.set_title(labels)
    ax.axis('off')
    plt.imshow(img)
  if display_only:
    plt.show()
  else:
    plt.savefig(os.path.join(save_folder, 'batch_{}.jpg'.format(i)))
 

# ———————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
  # NOTE: for image viewing to work, this currently has to be run on the devbox, not dgx
  root_dir = '/projects/pathology_char/pathology_char_results/mesothelioma/'
  tile_dir = os.path.join(root_dir, 'images')
  img_split = misc.load_pkl(os.path.join(root_dir, 
                              'results/6-2-2020/img_split.pkl'))
  img_to_label = misc.load_pkl(os.path.join(root_dir, 
                              'results/6-2-2020/img_to_label.pkl'))
  img_list = img_split['1']['train']

  dataset = SlideDataset(img_list, img_to_label, {'benign':0, 'tumor':1}, 
                               512, 16, transforms=transforms.ToTensor())

  save_folder = os.path.join(root_dir, 'dataloader_debugging')
  if not os.path.exists(save_folder): os.makedirs(save_folder)
  #for i in range(20):
  #  save_sample_imgs(save_folder, dataset)
  

  dataloader = create_dataloader('test', img_list, img_to_label, 
                                 label_to_value = {'benign':0., 'tumor':1.},
                                 img_size=512, batch_size=16, n_workers=1)
  sample_batch = next(iter(dataloader))

  view_from_dataloader(dataloader)
