import os
import sys
import argparse
import glob
from pathlib import Path
import time

from PIL import Image
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data

import misc
from pth_models import initialize_model

FLAGS = []

MAIN_DIR = '/path/to/dir/'
RUNS_MAIN = os.path.join(MAIN_DIR, 'results') 

# ——————————————————————————————————————————————————————————————————————
# model loading

def load_model_with_softmax(model_type, run_dir, epoch_weights=None):
  """ Loads a model and weights, adding a final softmax layer

  # Arguments
  
  # Returns: a pytorch model
  """
  # find checkpoint file and load model with weights
  ckpt_path = find_ckpt_path(run_dir, epoch_weights)
  print('loading model with weights from', ckpt_path)
  full_base_model = initialize_model(model_type, num_classes=2)
  full_base_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
  if model_type.startswith('resnet'):
    model = make_resnet_model(full_base_model)
  elif model_type.startswith('mobilenet'):
    model = make_mobilenet_model(full_base_model)
  elif model_type.startswith('inception'):
    model = make_inception_model(full_base_model)
  model.eval()
  return model

def make_resnet_model(full_base_model):
  # separate base and final fully connected layer
  layers = [x for x in full_base_model.children()]
  base = nn.Sequential(*layers[:-1])
  fc = layers[-1]  
  model = ResnetModel(base, fc)
  return model
   
# define and initiate custom model that will output multiple layers
class ResnetModel(nn.Module):
  def __init__(self, base, fc):
    super(ResnetModel, self).__init__()
    self.base = base
    self.fc = fc
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.base(x)
    base_out = x.reshape(x.size(0), -1) 
    fc_out = self.fc(base_out)
    softmax_out = self.softmax(fc_out)
    return softmax_out, fc_out, base_out

def make_mobilenet_model(full_base_model):
  layers = [x for x in full_base_model.children()]
  base = nn.Sequential(*layers[:-1])
  classifier = layers[-1]  
  model = MobilenetModel(base, classifier)
  return model
 
class MobilenetModel(nn.Module):
  def __init__(self, base, classifier):
    super(MobilenetModel, self).__init__()
    self.features = base
    self.classifier = classifier
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.features(x)
    x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
    base_out = x #x.reshape(x.size(0), -1) 
    fc_out = self.classifier(base_out)
    softmax_out = self.softmax(fc_out)
    return softmax_out, fc_out, base_out

def make_inception_model(full_base_model):
  # get named children and split names and features
  children = [x for x in full_base_model.named_children()]
  layer_names = [x[0] for x in children]
  layers = [x[1] for x in children]
  # remove auxiliary layers 
  base_layers = [x for i, x in enumerate(layers) 
                 if layer_names[i] != 'AuxLogits']
  base = nn.Sequential(*base_layers[:-1])
  fc = base_layers[-1]  
  model = InceptionModel(base, fc)
  return model

class InceptionModel(nn.Module):
  def __init__(self, base, fc):
    super(InceptionModel, self).__init__()
    self.base = base
    self.fc = fc
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    # need to split base to include pooling layers
    x = self.base[:3](x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = self.base[3:5](x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    x = self.base[5:](x)
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    base_out = torch.flatten(x, 1)
    fc_out = self.fc(base_out)
    softmax_out = self.softmax(fc_out)
    return softmax_out, fc_out, base_out


def find_ckpt_path(run_dir, epoch_weights=None):
  if epoch_weights is not None:
    ckpt_path = glob.glob(os.path.join(run_dir, 
                                       'ckpt*{}*pth'.format(epoch_weights))
                          )[0]
  elif os.path.exists(os.path.join(run_dir, 'ckpt_best_val.pth')):
    ckpt_path = os.path.join(run_dir, 'ckpt_best_val.pth')
  else:
    ckpt_list = glob.glob(os.path.join(run_dir, 'ckpt*pth'))
    ckpt_path = sorted(ckpt_list,
                       key=lambda x: int(os.path.splitext(os.path.basename(x)
                                                          )[0].split('_')[-1])
                       )[-1]
  return ckpt_path

# ——————————————————————————————————————————————————————————————————————
# prediction loop

class Dataset(data.Dataset):
  """ 
  Simple pytorch dataset to load images from a list and return 
  both the image and filename
  """
  def __init__(self, img_list, transforms):
    self.img_list = img_list
    self.transforms = transforms

  def __len__(self):
    return(len(self.img_list))

  def __getitem__(self, idx):
    img_path = self.img_list[idx]
    handle = os.path.splitext(os.path.basename(img_path))[0]

    img = Image.open(img_path)
    img = self.transforms(img)
    return img, handle


def make_dataloader(img_list, batch_size=32, n_workers=4):
  """ Simple function to make dataloader without any transforms
  """
  data_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225]),
                                       ])
  dataset = Dataset(img_list, data_transforms)
  dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           sampler=torch.utils.data.SequentialSampler(dataset),
                           pin_memory=True, num_workers=n_workers)
  return dataloader


def predict_on_images(case, model, dataloader, device, img_output_dir, 
                      case_output_dir, **kwargs):
  start_time = time.time()
  model = model.to(device)
  output_dict = {}
  output_by_image = {}
  handles_list = []

  for i, batch in enumerate(dataloader):
    if len(batch) == 2:
      # dataloader batches are a list of 1 item, with that containing 
      # a list of 2 items, the first of which is images and the 
      # second is labels
      inputs = batch[0]
      handles = batch[1]
      batch_size = len(inputs)
    else:
      inputs = batch
      batch_size = len(inputs)
    inputs = inputs.to(device)
    
    # get output from network and detach from gpu
    with torch.set_grad_enabled(False):
      softmax, fc, fv = model(inputs)
    softmax = softmax.cpu().numpy()
    fc = fc.cpu().numpy()
    fv = fv.cpu().numpy()
     
    handles_list += handles
    
    # concatenate each output item to output dictionary
    # not matched to an image patch)
    for output in ['softmax', 'fc', 'fv']:
      try:
        output_dict[output] = np.concatenate(
                                       (output_dict[output], 
                                        locals()[output]), 
                                       axis=0)
      except (NameError, ValueError, KeyError):
        output_dict[output] = locals()[output]
    
    for idx, handle in enumerate(handles):
      output_by_image[handle] = {'softmax': softmax[idx],
                                 'fc': fc[idx],
                                 'fv': fv[idx],}

  misc.save_pkl(output_dict, 
                    os.path.join(case_output_dir, case + '.pkl'))
  
  misc.save_pkl(output_by_image, 
                    os.path.join(img_output_dir, case + '.pkl'))
 
  str_ = '{:.02} min; fv {}'.format((time.time() - start_time)/60,
                                     output_dict['fv'].shape)
  return str_

# ——————————————————————————————————————————————————————————————————————

def process_cross_val(cv_idx, device, subdir, case_split, config,
                      image_main, dataloader_kwargs, epoch_weights=None,
                      **kwargs):

  case_output_dir = (os.path.join(subdir, 'predictions_cv', 'case'))
  img_output_dir = (os.path.join(subdir, 'predictions_cv', 'image'))
  
  for dir in [case_output_dir, img_output_dir]:
    Path(dir).mkdir(parents=True, exist_ok=True)

  completed_cases = [os.path.splitext(x)[0] 
                     for x in os.listdir(case_output_dir)]

  # load model
  model_type = config['model_type']
  model = load_model_with_softmax(model_type, subdir, epoch_weights)
  
  # loop over train/test and all cases
  for mode in ['train', 'test']:
    print('\n\n{}'.format(mode.upper()))

    # make case list
    try:
      split = case_split[str(cv_idx)][mode]
    except:
      split = case_split[mode]
    case_list = [x for x in split if x not in completed_cases]
    if FLAGS.benchmark_mode:
      case_list = case_list[:5]
    
    # set image directories to use
    img_subdirs = ['benign', 'tumor_full'] 
    img_subdirs = [os.path.join(image_main, x) for x in img_subdirs]
    print('using images from:', img_subdirs)  

    for i, case in enumerate(case_list):
      # compile list of images for each case
      # *** note that for prediction, only UNANNOTATED tumor images
      # are used ***
      img_list = []
      for img_subdir in img_subdirs:
        case_stem = os.path.join(img_subdir, case + '_*')
        img_list.extend(glob.glob(case_stem))
      
      # make dataloader
      dataloader = make_dataloader(img_list, **dataloader_kwargs)
      # generate feature vectors and probabilities for each tile
      str_ = predict_on_images(case, model, dataloader, device,
                               img_output_dir, case_output_dir)
 
      print('{}/{}: {} - {} imgs; {}'.format(i+1, len(case_list),
                                             case, len(img_list), str_))

def process_test_set(cv_idx, device, subdir, case_split, config, image_main,
                     dataloader_kwargs, output_dir, img_subdirs=None, 
                     epoch_weights=None, **kwargs):

  case_output_dir = (os.path.join(subdir, output_dir, 'case'))
  img_output_dir = (os.path.join(subdir, output_dir, 'image'))
  
  for dir in [case_output_dir, img_output_dir]:
    Path(dir).mkdir(parents=True, exist_ok=True)
 
  # load model
  model_type = config['model_type']
  model = load_model_with_softmax(model_type, subdir, epoch_weights)
 
  # make case list
  completed_cases = [os.path.splitext(x)[0] 
                     for x in os.listdir(case_output_dir)]
  case_list = [x for x in case_split if x not in completed_cases]
  img_subdirs = [os.path.join(image_main, x) for x in 
                 list(img_subdirs.values())]
  print('using images from:', img_subdirs)  
  for i, case in enumerate(case_list):
    # compile list of images for each case
    img_list = []
    for img_subdir in img_subdirs:
      # cases in referral set are given without block  e.g. BM_spin_VR16_19, 
      # while external set are given with block  e.g. ext_benign_P_16_5
      # need to include underscore for referrals or else there will be 
      # some overlap, such as for BM_spin_VR16_16 and BM_spin_VR16_1640
      if len(case.split('_')) == 4:
        extension = '_*'
      elif len(case.split('_')) == 5:
        extension = '*'
      case_stem = os.path.join(img_subdir, case + extension)
      img_list.extend(glob.glob(case_stem))
    
    # make dataloader
    dataloader = make_dataloader(img_list, **dataloader_kwargs)
    # generate feature vectors and probabilities for each tile
    str_ = predict_on_images(case, model, dataloader, device,
                             img_output_dir, case_output_dir)

    print('{}/{}: {} - {} imgs; {}'.format(i+1, len(case_list),
                                           case, len(img_list), str_))



# ——————————————————————————————————————————————————————————————————————
def main(FLAGS):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('running on device:', device)

  # anything that can be done from main run results folder is here,
  # while tasks that are cross val run specific (e.g. loading models),
  # are in the run processing functions

  run_dir = os.path.join(RUNS_MAIN, FLAGS.run_id)
  config_file = os.path.join(run_dir, 'setup_logs.txt')
  config = misc.parse_config_file(config_file)

  # load slide split
  case_split = misc.load_pkl(os.path.join(run_dir, 'case_split.pkl'))
  run_subdirs = glob.glob(os.path.join(run_dir, 'run_*'))

  # get main image dir from config unless not present
  try:
    image_main = os.path.dirname(config['benign_dir'].rstrip('\/'))
  except:
    image_main = FLAGS.image_main
  
  from folder_config import folder_config
  folder_config = folder_config.get(FLAGS.test_set, 'train')
 
  kwargs = {'image_main': image_main,
            'device': device,
            'config': config,
            'img_subdirs': folder_config['image_subdirs'],
            'case_split': case_split,
            'dataloader_kwargs': {'n_workers': FLAGS.n_workers,
                                  'batch_size': config['batch_size'],},
            'output_dir': folder_config['output_dir'],
            'epoch_weights': FLAGS.epoch_weights,
    }

  # run for each of the splits
  if FLAGS.test_set is not None:
    test_list = folder_config['list']
    kwargs['case_split'] = misc.load_txt_file_lines(test_list) 
    for i in range(1, len(run_subdirs)+1):
      print('\nbeginning run {}'.format(i))
      kwargs['subdir'] = run_subdirs[i-1]
      process_test_set(cv_idx=i, **kwargs)   
  elif FLAGS.cv_idx == None:
    for i in range(1, len(run_subdirs)+1):
      print('\nbeginning run {}'.format(i))
      kwargs['subdir'] = run_subdirs[i-1]
      process_cross_val(cv_idx=i, **kwargs)
  else:
    kwargs['subdir'] = os.path.join(run_dir, 'run_{}'.format(FLAGS.cv_idx))
    process_cross_val(FLAGS.cv_idx, **kwargs)
  print('\nPREDICTIONS COMPLETE')
  sys.exit(0)

# ——————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_main', default=None,
      help='full path to main image directory at given resolution')
  parser.add_argument('--run_id',
      help="Saved model to use for generating predictions")  
  parser.add_argument('--cv_idx', default=None, type=int)

  parser.add_argument('--img_size', type=int, default=512,
      help='Side dimension for tiles')
  parser.add_argument('--epoch_weights', default=None, type=int,
      help='Train epoch from which saved weights should be loaded.')
  parser.add_argument('--n_workers', default=6, type=int)
  parser.add_argument('--benchmark_mode', action='store_true')

  # testing related options
  parser.add_argument('--test_set', default=None,
      choices=['external', 'referrals'])
  
  FLAGS = parser.parse_args()
  print(FLAGS)
  main(FLAGS)
