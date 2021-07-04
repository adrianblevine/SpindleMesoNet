""" Makes the cross-validation image split and runs training

Assumes that slides are split in a single folder for each category
"""

import os
import glob
import sys
import argparse
import random
import datetime
import time
import statistics

import numpy as np
from PIL import Image
import torch

import misc
import trainer
import dataloader

FLAGS = []

# —————————————————————————————————————————————————————————————————————————————
# configuration

slide_formats = ['svs', 'tif', 'tiff']
img_formats = ['jpg', 'tif', 'tiff', 'png']

main_dir = '/path/to/dir/'

categories = ['benign', 'tumor']

slide_dirs = {
  'benign': '/path/to/dir',
  'tumor': '/path/to/dir',
  }

slide_lists = {
  'benign':'/path/to/dir',
  'tumor': '/path/to/dir',
  }

results_main = os.path.join(main_dir, 'results')

# —————————————————————————————————————————————————————————————————————————————
# functions to load labels and generate the train/val/test split

def load_text_file_to_list(filepath):
  with open(filepath, 'r') as f:
    list_ = f.read().splitlines()
  return list_

def write_list_to_text_file(list_, filepath):
  with open(filepath, 'w') as f:
    for item in list_:
      f.write("{}\n".format(item))

def slide_to_case(x):
  return '_'.join(x.split('_')[:4]) 

def verify_splits(case_split, slide_split, img_split):
  # print allocation of images, slides, and cases to train/test
  print('case allocation:', [(key, len(case_split[key])) 
                                for key in case_split.keys()])
  print('slide allocation:', [(key, len(slide_split[key])) 
                                for key in slide_split.keys()])
  print('image allocation:', [(key, len(img_split[key])) 
                                for key in img_split.keys()])
  # *** verify that there is no overlap between train and test sets ***
  print('train/test overlap between cases, slides, and images:', 
      list(set(case_split['train']) 
               & set(case_split['test'])),
      (list(set(slide_split['train']) & set(slide_split['test']))),
      len(list(set(img_split['train']) & set(img_split['test']))),)
  

def make_case_split(slide_dirs, image_dirs, results_dir, cross_val=None, 
                     train_list=None, test_list=None, val_proportion=0.2, 
                     n_test=0.33):
  """ Generates train/test case split

  Will not split a single case within multiple sets. 

  # Args:
    slide_dir: path to main slide directory
    n_test: number of cases allotted to training; can either be an integer 
            for the absolute number or a fraction for the proportion of 
            slides allotted to each
  
  # Returns: two dictionaries
  """

  # make list of categories, image patches, slides, and cases
  image_dict = {category: [im for im in os.listdir(image_dirs[category]) 
                           if misc.verify_is_image(im)] 
                for category in categories}

  image_paths = {category: [im for im in glob.glob(image_dirs[category] + '/*')
                            if misc.verify_is_image(im)] 
                 for category in categories}
  
  # make list of necrosis and normal lung images - to add to train set ONLY
  images_main = os.path.dirname(image_dirs['benign'].rstrip('/'))
  lung_imgs = glob.glob(os.path.join(images_main, 'lung', '*png'))
  print('found {} normal lung images'.format(len(lung_imgs)))
  
  # slide lists are loaded from specified files and cases are
  # identified from slide lists; this works better than determining slides
  # from saved images (which can miss some slides) or from slide folders
  # (some tumor slides aren't annotated)
  slide_dict = {category: load_text_file_to_list(slide_lists[category])
                for category in categories}

  case_dict = {category: list(set(['_'.join(x.split('_')[:4]) 
                            for x in slide_dict[category]])) 
                  for category in categories}
  
  # identify cases without training images
  zero_imgs = []
  for category in categories:
    for case in case_dict[category]:
      n_imgs = len([x for x in image_dict[category] if case in x])
      if n_imgs == 0:
        zero_imgs.append(case)
  print('\ncases with zero train images: {} \n'.format(zero_imgs))

  # make dictionaries mapping each image/slide to its label
  img_to_label, slide_to_label = {}, {}
  for category in categories:
    for img in image_dict[category]:
      img_to_label[img] = category 
    for slide in os.listdir(slide_dirs[category]):
      slide_to_label[slide.split('.')[0]] = category
  
  # print number of patinets, slides, and images per category and
  # any cases with slides in multiple categories
  for c in categories:
    print('{}: {} cases, {} slides, {} images'. format(c, 
          len(case_dict[c]), len(slide_dict[c]), len(image_dict[c])))
  overlap = list(set(case_dict['benign']) & set(case_dict['tumor']))
  print('cases with both benign and tumor slides:', overlap)

  # make split using specified test list
  if test_list is not None:
    test_list = [os.path.splitext(x)[0] for x in test_list]
    all_cases = []
    for k in case_dict.keys():
      all_cases.extend(case_dict[k])
    
    if train_list is None:
      train_list = [x for x in all_cases if x not in test_list]

    case_split = {'train': train_list,
                    'test': test_list}
    slide_split = {'train': [], 'test': []}
    img_split = {'train': [], 'val': [],'test': []}
     
    for c in categories:
      for mode in ['test', 'train']:
        # loop over each case in the split
        for case in case_split[mode]:
          # add slides to slide split
          slides = [x for x in slide_dict[c] if case + '_' in x]
          slide_split[mode].extend(slides)
          # add images to train or train/val
          imgs = [x for x in image_paths[c] if case + '_' in x]
          if mode == 'train' and val_proportion is not None:
            random.shuffle(imgs)
            n_val = int(val_proportion * len(imgs))
            img_split['val'] += imgs[:n_val]
            img_split['train'] += imgs[n_val:]
          else:
            img_split[mode] += imgs
    # add normal lung images to train list
    img_split['train'] += lung_imgs
    verify_splits(case_split, slide_split, img_split)

  # make random multi-fold cross validation split
  elif cross_val is not None:
    # set up dictionaries for cases, slides, and images
    # train/validation image splits are both from train cases and
    # there is case overlap between these sets
    case_split = {'{}'.format(x+1): {y: [] for y in ['train', 'test']} 
                     for x in range(cross_val)}
    slide_split = {'{}'.format(x+1): {y: [] for y in ['train', 'test']} 
                     for x in range(cross_val)}
    img_split = {'{}'.format(x+1): {y: [] for y in ['train', 'val', 'test']} 
                     for x in range(cross_val)}
 
    for category in categories:
      random.shuffle(case_dict[category])
      n_cases = len(case_dict[category])
      for i in range(cross_val):
        split_idx = str(i+1)
        start_idx = int(n_cases * i/cross_val)
        end_idx = int(n_cases * (i + 1)/cross_val)
        test_list = case_dict[category][start_idx: end_idx]
        case_split[split_idx]['test'] += test_list 
        case_split[split_idx]['train'] += [x for x in case_dict[category] 
                                              if x not in test_list]

        for mode in ['test', 'train']:
          for case in case_split[split_idx][mode]:
            slides = [x for x in slide_dict[category] if case + '_' in x] 
            slide_split[split_idx][mode].extend(slides)
            imgs = [x for x in image_paths[category] if case + '_' in x] 
            
            if mode == 'train' and val_proportion is not None:
              random.shuffle(imgs)
              n_val = int(val_proportion * len(imgs))
              img_split[split_idx]['val'] += imgs[:n_val]
              img_split[split_idx]['train'] += imgs[n_val:]
            else:
              img_split[split_idx][mode] += imgs
        # add lung images to benign set
        img_split[split_idx]['train'] += lung_imgs
    
    for k in case_split.keys():
      print('\nsplit {}:'.format(k))
      verify_splits(case_split[k], slide_split[k], img_split[k])
       
  # make single random split
  else:
    if n_test < 1:
      n_test = int(len(cases)*n_test)
    one_slide = [x for x in case_dict.keys() if len(case_dict[x]) == 1]
    random.shuffle(one_slide)
    case_split = {'test': one_slide[:n_test]}
    case_split['train'] = [x for x in cases if x not in case_split['test']]
                 
    slide_split = {}
    for mode in ['test', 'train']:
      slide_split[mode] = []
      for case in case_split[mode]:
        slide_split[mode] += [os.path.join(slide_dir, x) 
                              for x in case_dict[case]] 
    verify_splits(case_split, slide_split, img_split)
  
  # save all relevant dictionaries
  misc.save_pkl(case_split, os.path.join(results_dir, 'case_split.pkl'))
  misc.save_pkl(img_split, os.path.join(results_dir, 'img_split.pkl'))
  misc.save_pkl(slide_split, os.path.join(results_dir, 'slide_split.pkl'))
  misc.save_pkl(img_to_label, os.path.join(results_dir, 'img_to_label.pkl'))
  misc.save_pkl(slide_to_label, os.path.join(results_dir, 'slide_to_label.pkl'))
  return img_split, img_to_label, case_split  

# —————————————————————————————————————————————————————————————————————————————

def _main(FLAGS):
  dt = datetime.datetime.now() 

  image_dirs = {'benign': FLAGS.benign_dir,
                'tumor': FLAGS.tumor_dir}
  
  run_id = (FLAGS.run_id if FLAGS.run_id 
            else '{}-{}-{}'.format(dt.day, dt.month, dt.year))
  results_dir = os.path.join(results_main, run_id)
  misc.verify_dir_exists(results_dir)

  misc.init_output_logging(os.path.join(results_dir, 'setup_logs.txt'))
  print(FLAGS)
  print('training on {} gpu(s)'.format(torch.cuda.device_count()))
  print('saving to:', results_dir)
  
  # load slide split if there is one saved, otherwise make it
  if (os.path.exists(os.path.join(results_dir, 'img_split.pkl'))
      and not FLAGS.redo_split):
    img_split = misc.load_pkl(os.path.join(results_dir, 'img_split.pkl'))
    case_split = misc.load_pkl(os.path.join(results_dir, 
                                                'case_split.pkl'))
    img_to_label = misc.load_pkl(os.path.join(results_dir, 'img_to_label.pkl'))
  
  # make split based on specified files for train and test lists
  elif FLAGS.test_list:
    try:
      train_list = load_text_file_to_list(FLAGS.train_list)    
    except:
      train_list=None
    test_list = load_text_file_to_list(FLAGS.test_list)    
    img_split, img_to_label, case_split = make_case_split(slide_dirs, 
                                                image_dirs, results_dir,
                                                train_list=train_list,
                                                test_list=test_list)
  # make random splits
  else:
    img_split, img_to_label, case_split = make_case_split(slide_dirs, 
                                                image_dirs, results_dir,
                                                FLAGS.cross_val)
  misc.end_logger()
  
  # dataloader and trainer parameters
  dataloader_kwargs = {'img_size': FLAGS.img_size, 
                       'batch_size': FLAGS.batch_size, 
                       'n_workers': FLAGS.n_cpu, 
                       'label_to_value': {'benign': 0., 'tumor': 1.},
                       'color_jitter': FLAGS.color_jitter,
                       'profiler_mode': FLAGS.profiler_mode,
                      }

  trainer_kwargs = {'profiler_mode': FLAGS.profiler_mode,
                    'model_type': FLAGS.model_type, 
                    'n_epochs': FLAGS.n_epochs, 
                    'input_size': FLAGS.img_size,}
  
  # maximum number of images to use per epoch per case
  max_imgs = {'train': 1500, 'val': 10000, 'test': 10000}
  
  start_time = time.time()
  
 
  if FLAGS.cross_val is not None:
    n_split = 1 if FLAGS.profiler_mode else FLAGS.cross_val
    test_accs = [] # compile all test accurcies
    
    # make dataloaders and run training for each cross-val split 
    for i in range(1, n_split + 1):
      print('\n\nRUN: {}'.format(i))
      run_split = img_split[str(i)]
      results_subdir = os.path.join(results_dir, 'run_{}'.format(i))
      misc.verify_dir_exists(results_subdir)

      dataloaders = {}
      dataloaders['train'] = {}
      for i in range(1, FLAGS.n_epochs+1):
         dataloaders['train'][i] = dataloader.create_dataloader(phase='train', 
                                                   img_list=run_split['train'], 
                                                   img_to_label=img_to_label,
                                                   export_dir=results_subdir,
                                                   max_imgs=max_imgs['train'],
                                                   **dataloader_kwargs)
      print('dataset length:', len(dataloaders['train'][1].dataset))

      for phase in ['val', 'test']:
        dataloaders[phase] = dataloader.create_dataloader(phase=phase, 
                                                    img_list=run_split[phase], 
                                                    img_to_label=img_to_label,
                                                    export_dir=results_subdir,
                                                    max_imgs=max_imgs[phase],
                                                    **dataloader_kwargs)

      model_trainer = trainer.Trainer(dataloaders, results_subdir, 
                                      **trainer_kwargs)
      test_acc = model_trainer.train_main(optimizer=FLAGS.optimizer)
      test_accs.append(test_acc)

    print('\nmean test accuracy: {:.03} (std {:.02})'.format(
                                                  statistics.mean(test_accs),
                                                  statistics.stdev(test_accs)))
  else: 
      dataloaders = {}
      results_subdir = os.path.join(results_dir, 'run_1')
      for phase in ['train', 'val', 'test']:
        dataloaders[phase] = dataloader.create_dataloader(
                                                    phase=phase, 
                                                    img_list=img_split[phase], 
                                                    img_to_label=img_to_label,
                                                    export_dir=results_subdir,
                                                    **dataloader_kwargs)

      model_trainer = trainer.Trainer(dataloaders, results_subdir, 
                                      **trainer_kwargs)
      test_acc = model_trainer.train_main(optimizer=FLAGS.optimizer)

  sys.exit(0)

# —————————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--run_id', default=None)
  parser.add_argument('--batch_size', type=int, default=64,
      help="Depends on gpu type and number, and on model. Suggestions: "
           "resnet18: dev 2: ; dgx 1: "
           "mobilenet: dev 2: 16; dgx 1:")
  parser.add_argument('--img_size', type=int, default=512)
  parser.add_argument('--n_epochs', type=int, default=6)
  parser.add_argument('--n_cpu', type=int, default=8)
  
  parser.add_argument('--model_type', default='resnet18')
  parser.add_argument('--optimizer', choices=['adam', 'sgdm'], default='sgdm')  
  
  parser.add_argument('--cross_val', type=int, default=None) 
  parser.add_argument('--redo_split', action='store_true', default=False,
      help='redo train/test split generation regardless of whether one exists') 
  
  parser.add_argument('--color_jitter', action='store_true', 
      default=False)
  parser.add_argument('--benign_dir', default='images_40x/benign')
  parser.add_argument('--tumor_dir', default='images_40x/tumor_annotated_1')
  parser.add_argument('--train_list', default=None)
  parser.add_argument('--test_list', default=None)
  
  parser.add_argument('--profiler_mode', action='store_true', default=False)
   
  FLAGS = parser.parse_args()
  
  for f in ['benign_dir', 'tumor_dir', 'train_list', 'test_list']:
    try:
      arg = vars(FLAGS)[f]
      if not os.path.isabs(arg):
        vars(FLAGS)[f] = os.path.join(main_dir, arg)
    except TypeError: pass
 
  _main(FLAGS)
