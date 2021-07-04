import os
import sys
import argparse
import glob
import pickle
import random
from multiprocessing import Pool
import functools
import time
import itertools
import warnings
import statistics
import csv 

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, decomposition

from slide_prediction import (EvaluatePredictions, get_label_from_filename, 
  print_predictions, plot_roc_curve_test, plot_roc_curve_crossval)

try:
  import torch
  import torch.nn as nn
  import torch.utils.data as data
  import torch.optim as optim
  import torch.backends.cudnn as cudnn
  from trainer import running_metrics
  torch_available=True
except ImportError:
  print('unable to import torch')
  torch_available=False

import misc


FLAGS = []

main_dir = '/path/to/dir/'

# ——————————————————————————————————————————————————————————————————————
# helper functions

label_to_value = {'benign': 0.,
                  'tumor': 1.}


def np_to_str(x):
  return np.array_str(x)[1:-1]


def _tile_in_region(tile, region, xmin, xmax, ymin, ymax):
  if os.path.isabs(tile):
    tile = os.path.basename(tile)
  coords = misc.get_coords(tile)
  if (os.path.basename(tile).split('-')[0] == region.split('-')[0] and
      xmin <= coords[0] <=xmax) and (ymin <= coords[1] <= ymax):
    return True
  else:
    return False


def get_feature_vector_length(run_subdir):
  pkls = glob.glob(os.path.join(run_subdir, 
                                'predictions_cv/image', '*pkl'))
  preds = misc.load_pkl(pkls[0])
  data = preds[list(preds.keys())[0]]
  fv_length = data['fv'].shape[0]
  print('feature vector length:', fv_length)
  return fv_length


class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_run_folders(x, subdir):
  y = AttributeDict()
  for key_a in x.keys():
    value_a = x[key_a]
    if isinstance(value_a, dict):
      y[key_a] = AttributeDict()
      for key_b in value_a.keys():
        value_b = value_a[key_b]
        if os.path.isabs(value_b):
          y[key_a][key_b] = value_b
        else:
          y[key_a][key_b] = os.path.join(subdir, value_b)
    else:
      if os.path.isabs(value_a):
        y[key_a] = value_a
      else:
        y[key_a] = os.path.join(subdir, value_a)
  return y


def write_to_csv_file(csv_file, reset_csv):
  if reset_csv:
    if os.path.exists(csv_file): os.remove(csv_file)

  try:
    with open(csv_file) as f:
      n_lines = (len(f.readlines()))
  except: n_lines = 0

  with open(csv_file, 'a', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    # write headers if empty
    if n_lines == 0:
      header = ['run_id', 'pred_type', 'auc', 'acc', 'precision',
                'recall', 'f1']
      w.writerow(header)
    # write output to file
    for k in metrics.keys():
      line = [FLAGS.run_id, k] + [round(x, 5) for x in (metrics[k])]
      w.writerow(line)


def determine_best_threshold(thresholds, preds, labels):
  best_acc, best_value = 0, 0
  for value in thresholds:
    print('\n{}:'.format(value))
    # metrics returned: AUC, acc, precision, recall, F1
    metrics = evaluate_run(preds, labels, value)
    acc = metrics[1]
    if acc > best_acc:
      best_acc = acc
      best_value = value
  print('\n* best accuracy {:.03}; best threshold {} *'.format(best_acc,
                                                           best_value))
  return best_value


def print_run_separator(idx):
  print('\n===============================================================')
  print('RUN {}'.format(idx))
 

# ——————————————————————————————————————————————————————————————————————
# RNN training and prediction
if torch_available:
  class RNN_model(nn.Module):
    def __init__(self, input_size=512, ndims=64, apply_softmax=False):
      super(RNN_model, self).__init__()
      self.ndims = ndims
      self.apply_softmax = apply_softmax

      self.fc1 = nn.Linear(input_size, ndims)
      self.fc2 = nn.Linear(ndims, ndims)
      self.fc3 = nn.Linear(ndims, 2)

      self.activation = nn.ReLU()
      self.softmax = nn.Softmax(dim=1)

    def forward(self, input, state):
      input = self.fc1(input)
      state = self.fc2(state)
      state = self.activation(state+input)
      output = self.fc3(state)
      if self.apply_softmax:
        output = self.softmax(output)
      return output, state

    def init_hidden(self, batch_size):
      return torch.zeros(batch_size, self.ndims)


  class RNN_data(data.Dataset):
    def __init__(self, data_dir, item_list):
      super(RNN_data, self).__init__()
      self.data_dir = data_dir
      self.item_list = item_list

    def __getitem__(self, index):
      path = os.path.join(self.data_dir, self.item_list[index])
      data = misc.load_pkl(path)
      try:
        label = label_to_value[data['label']]
      except KeyError:
        label = label_to_value[get_label_from_filename(os.path.basename(path))]
      label = torch.LongTensor(np.array(label))
      inputs = data['inputs']
      return (inputs, label, self.item_list[index])

    def __len__(self):
      return len(self.item_list)


def train_loop(epoch, rnn, loader, criterion, optimizer, device):
  rnn.train()
  running_loss = 0.
  tp, tn, fp, fn = 0, 0, 0, 0

  for i,(inputs,label,_) in enumerate(loader):
    batch_size = inputs.size(0)
    len_seq = inputs.size(1)
    rnn.zero_grad()

    state = rnn.init_hidden(batch_size).to(device)
    for s in range(len_seq):
      input = inputs[:,s,:].to(device)
      output, state = rnn(input, state)

    label = label.to(device)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    _, pred = torch.max(output, 1)
     
    running_loss += loss.item()*batch_size
    acc, sens, spec, F1, tp, tn, fp, fn = running_metrics(pred, label,
                                          tp, tn, fp, fn)

  loss = running_loss/len(loader.dataset)
  metrics = {'loss': round(loss, 3),
             'acc': round(acc, 3),
             'sens': round(sens, 3),
             'spec': round(spec, 3),
             'F1': round(F1, 3),
             }  
  print('Train - Epoch: {}\t'.format(epoch+1), metrics, flush=True)
  return loss, acc, F1


def test_loop(epoch, rnn, loader, criterion, device, mode='Val'):
  rnn.eval()
  running_loss = 0.
  tp, tn, fp, fn = 0, 0, 0, 0

  with torch.no_grad():
    for i,(inputs,label,_) in enumerate(loader):
      batch_size = inputs.size(0)
      len_seq = inputs.size(1)
      
      state = rnn.init_hidden(batch_size).to(device)
      for s in range(len_seq):
        input = inputs[:,s,:].to(device)
        output, state = rnn(input, state)

      label = label.to(device)
      loss = criterion(output,label)
      _, pred = torch.max(output, 1)

      running_loss += loss.item()*batch_size
      acc, sens, spec, F1, tp, tn, fp, fn = running_metrics(pred, label,
                                                            tp, tn, fp, fn)
      try: 
        labels = np.concatenate((labels, label.cpu()))
        preds = np.concatenate((preds, pred.cpu()))
      except NameError:
        labels = np.array(label.cpu())
        preds = np.array(pred.cpu())
    
  loss = running_loss/len(loader.dataset)
  metrics = {'loss': round(loss, 3),
             'acc': round(acc, 3),
             'sens': round(sens, 3),
             'spec': round(spec, 3),
             'F1': round(F1, 3),
             }  
  print('{} - Epoch: {}\t'.format(mode, epoch+1), metrics, flush=True)
  _ = evaluate_predictions(preds, labels, threshold=0.5, print_output=True)

  return loss, acc, F1


def rnn_main(folders, fv_length, test_set='val', n_process=1,  **kwargs):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('\nRNN training (running on {}):'.format(device))
  
  # potential hyperparameters to search
  nepochs = kwargs.get('nepochs', 5)
  weights = kwargs.get('weights', 0.5)
  ndims = kwargs.get('ndims', 64)

  val_fraction = 0.1

  # make dataloaders
  dataloader_kwargs = {'batch_size': 128, 
                       # note: num_workers has to be 0 for pdb to work 
                       # (ie not 1)
                       'num_workers': n_process, 
                       'pin_memory': False}
  # train/val split
  train_data_dir = folders.rnn_data.train
  item_list = os.listdir(train_data_dir)
  random.shuffle(item_list)
  n_val = int(len(item_list) * val_fraction)
  train_list = item_list[n_val:] 
  val_list = item_list[:n_val]
 
  train_dset = RNN_data(train_data_dir, train_list)
  train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True,
                                             **dataloader_kwargs)
  val_dset = RNN_data(train_data_dir, val_list)
  val_loader = torch.utils.data.DataLoader(val_dset, shuffle=True,
                                             **dataloader_kwargs)

  test_data_dir = folders.rnn_data.test
  test_dset = RNN_data(test_data_dir, os.listdir(test_data_dir))
  test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False,
                                            **dataloader_kwargs)

  # make model
  rnn = RNN_model(input_size=fv_length, ndims=ndims)
  rnn = rnn.to(device)
  
  ## optimization
  # note that the CrossEntropyLoss expects raw, unnormalized scores 
  # (i.e. no softmax) and combines nn.LogSoftmax() and nn.NLLLoss() 
  # in one single class
  if weights==0.5:
    criterion = nn.CrossEntropyLoss().to(device)
  else:
    w = torch.Tensor([1-weights,weights])
    criterion = nn.CrossEntropyLoss(w).to(device)
  # parameters from Fuchs paper: 0.1, momentum=0.9, dampening=0, 
  #                              weight_decay=1e-4, nesterov=True 
  optimizer = optim.SGD(rnn.parameters(), 0.01, momentum=0.9, dampening=0, 
                        weight_decay=kwargs.get('weight_decay', 0.1), 
                        nesterov=True)
  cudnn.benchmark = True

  with open(os.path.join(folders.subdir, 'rnn_convergence.csv'), 'w') as fconv:
    fconv.write('epoch,train.loss,train.fpr,train.fnr,'
                 'val.loss,val.fpr,val.fnr\n')
  best_acc = 0.
  for epoch in range(nepochs):
    train_loss, train_acc, train_F1 = train_loop(epoch, rnn, train_loader, 
                                                 criterion, optimizer, 
                                                 device)
    val_loss, val_acc, val_F1 = test_loop(epoch, rnn, val_loader, 
                                           criterion, device)
    with open(os.path.join(folders.subdir,'rnn_convergence.csv'), 'a') as fconv:
      fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, 
                                                train_acc, train_F1, 
                                                val_loss, val_acc, val_F1))

    if val_acc > best_acc or epoch==0:
      best_acc = val_acc
      obj = {
        'epoch': epoch+1,
        'state_dict': rnn.state_dict()
      }
      torch.save(obj, os.path.join(folders.subdir,'rnn_checkpoint_best.pth'))
  obj = torch.load(os.path.join(folders.subdir,'rnn_checkpoint_best.pth'),                                           map_location='cpu')
  rnn.load_state_dict(obj['state_dict'])
  _, _,_, = test_loop(0, rnn, test_loader, criterion, device, 
                      mode='Test')

def rnn_prediction(folders, case_split, fv_length, save_incorrect=False, 
                   n_process=1, **kwargs):
  print('\nRunning RNN predictions:')

  # initiate RNN and load checkpoint
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ndims = kwargs.get('ndims', 64)
  rnn = RNN_model(input_size=fv_length, ndims=ndims, apply_softmax=True)
  rnn.to(device)
  rnn.eval()
  
  obj = torch.load(os.path.join(folders.subdir,'rnn_checkpoint_best.pth'),                                 map_location='cpu')
  rnn.load_state_dict(obj['state_dict'])
  n_saved = 0

  # detect modes (ie train and val/test) directly from split to
  # accommodate either cross validation or a separate test set
  for mode in case_split.keys():
    item_list = os.listdir(folders.rnn_data[mode])
    output_dir = folders.rnn_preds[mode]
    misc.verify_dir_exists(output_dir, remove_existing=False)

    for case in case_split[mode]:
      regions = [x for x in item_list if case in x]
      dataloader_kwargs = {'batch_size': 128, 
                           'num_workers': n_process, 
                           'pin_memory': False, 
                           'shuffle': False}
      dset = RNN_data(folders.rnn_data[mode], regions)
      loader = torch.utils.data.DataLoader(dset, **dataloader_kwargs)
      # reset predictions
      preds = np.empty((0,0))
      with torch.no_grad():
        for i,(inputs,label,ids) in enumerate(loader):
          batch_size = inputs.size(0)
          len_seq = inputs.size(1)
          
          state = rnn.init_hidden(batch_size).to(device)
          for s in range(len_seq):
            input = inputs[:,s,:].to(device)
            output, state = rnn(input, state)
          pred = output[:, 1]
          try:
            preds = np.concatenate((preds, pred.cpu()))
          except (ValueError, NameError):
            preds = pred.cpu()
          
      misc.save_pkl(preds, os.path.join(output_dir, case + '.pkl'))
  if save_incorrect:
    print('saved {} incorrect images to:'.format(n_saved), folders.img_save_dir)

# ——————————————————————————————————————————————————————————————————————
# RNN data preprocessing


class RNNPreprocessor():
  def __init__(self, folders, case_split, n_process, test_set):
    """
    # Args
    subdir: run subdirectory, e.g. /path/to/main/run_1
    case_split: train/test split (either cross val or true test set)
    cnn_pred_dirs: image level prediction dir 
    n_process
    """
    self.case_split = case_split
    self.test_set = test_set
    self.n_process = n_process
    self.subdir = folders.subdir
    self.cnn_pred_dirs = folders.cnn_preds
    self.rnn_data_dirs = folders.rnn_data

    # make lists of regions
    #TODO
    regions = {x: os.listdir(os.path.join(folders.region_img_dir, x))
               for x in ['benign', 'tumor_full', 'tumor_annotated']}
    benign_dir = os.path.join(folders.region_img_dir, 
                              folders.test_subdirs['benign'])
    tumor_dir = os.path.join(folders.region_img_dir, 
                             folders.test_subdirs['tumor']) 
    test_regions = (os.listdir(benign_dir) + os.listdir(tumor_dir))
    print('\nnumber of regions: benign {}; tumor {}, test {}'.format(
          len(regions['benign']), len(regions['tumor_full']), 
          len(test_regions)))

    self.regions = {'train': regions['benign'] + regions['tumor_annotated'],
                    'val':  regions['benign']  + regions['tumor_full'],
                    'test':  test_regions}
    
    # load list of tiles
    try:
      self.tiles_list = misc.load_pkl(os.path.join(FLAGS.tile_dir, 
                                                   'tiles_list.pkl'))
    except:
      #TODO
      self.tiles_list = (
               glob.glob(os.path.join(FLAGS.tile_dir, 'benign') + '*png') 
             + glob.glob(os.path.join(FLAGS.tile_dir, 'tumor_full', '*png'))) 
      misc.save_pkl(self.tiles_list, os.path.join(FLAGS.tile_dir, 
                                                  'tiles_list.pkl'))
  
  def preprocess_rnn_data(self, mode):
    print('\n{}'.format(mode.upper()), flush=True)
    input_dir = self.cnn_pred_dirs[mode]
    pkl_list = glob.glob(os.path.join(input_dir, '*.pkl'))
    save_dir = self.rnn_data_dirs[mode]
    misc.verify_dir_exists(save_dir, remove_existing=False)
    case_list = self.case_split[mode]
    
    if self.n_process > 1:
      pool = Pool(self.n_process)
      function = functools.partial(self.process_case, pkl_list=pkl_list, 
                                   save_dir=save_dir, mode=mode) 
      pool.map_async(function, case_list)
      pool.close(); pool.join()
    
    # single process for debugging
    else:
      for case in case_list:
        self.process_case(case, pkl_list, save_dir, mode)

  def process_case(self, case, pkl_list, save_dir, mode, use_full_case=True):
    # make list of regions for case
    case_regions = [x for x in self.regions[mode] if case in x]
    # load predictions for all tiles 
    preds_pkl = [x for x in pkl_list if case in os.path.basename(x)][0]
    case_preds = misc.load_pkl(preds_pkl)
    
    start_time = time.time()
    if len(case_regions) > 0:
      n_errors = 0
      for region in case_regions:
        try:
          self.get_feature_vectors(region=region, case_preds=case_preds, 
                                   save_dir=save_dir, by_region=True)   
        except KeyError:
          n_errors += 1
      if n_errors > 0:
        print('* case {} errors: {}'.format(case, n_errors))        
    elif use_full_case:
      # aggregate feature vectors for entire case if no regions
      # (determined using the by_region variable in get_feature_vectors)
      try:
        self.get_feature_vectors(region=case, case_preds=case_preds, 
                                 save_dir=save_dir, by_region=False)   
        print('no regions for case {}, aggregating vectors from '
              'whole case'.format(case))
      except KeyError as e:
        print(case, e, flush=True)   
    else:
      pass
    time_taken = round((time.time() - start_time),2)
    print('{} ({}): {} s'.format(case, len(case_regions), time_taken), 
          flush=True)

  def get_feature_vectors(self, region, case_preds, save_dir, topN=10,
                          by_region=True, dimensionality_reduction=False):
    """ Get top feature vectors for a given region
    
    # Args:
    region: region to process
    case_preds: dict containing predictions for all images for the case
    save_dir: rnn data dir
    topN: number of feature vectors to use for region
    by_region: if false will aggregate feature vectors for a given case
        (i.e. use if case has no regions)
    dimensionality_reduction: optional PCA dimensionality reduction on all
        feature vectors

    """
    if by_region:
      # make list of pkl files for given region
      xmin, ymin = misc.get_coords(region)
      xmax = xmin + 5120 - 512
      ymax = ymin + 5120 - 512
      region_imgs = [x for x in list(case_preds.keys())
                     if _tile_in_region(os.path.basename(x), region, xmin, 
                                        xmax, ymin, ymax)]
      assert len(region_imgs) < 101
    else:
      region_imgs = case_preds

    pred_dict = {}
    # get label
    label = get_label_from_filename(region)
    # load pkls
    for img in region_imgs:
      pred = case_preds[img] 
      for output in pred.keys():
        data = pred[output].reshape((1,-1))
        try:
          pred_dict[output] = np.concatenate((pred_dict[output],
                                               data))
        except (NameError, ValueError, KeyError):
          pred_dict[output] = data
    
    softmax = pred_dict['softmax'][:,1]
    softmax = softmax.reshape((-1,1))
    fv = pred_dict['fv']
    
    z = np.array(list(zip(softmax,fv)))
    # rank feature vectors by softmax value and unzip
    z = z[z[:,0].argsort()]
    softmax, full_fv = zip(*(tuple(z)))
    
    if len(full_fv) >= 10:
      # take top N feature vectors
      full_fv = np.array(full_fv)
      fv = full_fv[-topN:, :]
      
      # reshape and concatenate full fv array
      save_dict = {'inputs': fv,
                   'label': label,} 
      save_path = (os.path.join(save_dir, region))
      misc.save_pkl(save_dict, save_path)    
    else:
      print('excluding region {} ({} FVs)'.format(region, len(full_fv)))

def _check_alignment(x, y):
  print('*****************************')
  for i in range(len(x)):
    print(x[i], y[i][0])

# ——————————————————————————————————————————————————————————————————————
# slide level prediction using output from RNN

class SlidePrediction():
  def __init__(self, folders, case_split):
    self.subdir = folders.subdir
    self.case_split = case_split
    self.rnn_preds = folders.rnn_preds
    # use tile predictions not separated by individual image because it's
    # either to then pull softmax values
    self.cnn_preds_dir = folders.cnn_preds
    self.histogram_dir = os.path.join(folders.subdir, 'rnn_histograms')
    misc.verify_dir_exists(self.histogram_dir, remove_existing=True)
    self.load_region_predictions()

  def load_region_predictions(self):
    """ Load predictions for all regions for a given case
    """ 
    self.output_dict = {'train': {}, 'test':{} }
    for mode in ['train', 'test']:
      pred_dir = self.rnn_preds[mode]
      pkl_list = [x for x in glob.glob(pred_dir + '/*') if x.endswith('pkl')]
      for pkl_path in pkl_list: 
        case_id = os.path.splitext(os.path.basename(pkl_path))[0]
        try:
          pred =  misc.load_pkl(pkl_path)
          
          if len(pred) == 0:
            # if no regions load top 500 tile level predictions
            dir_ = os.path.split(self.cnn_preds_dir[mode])[0]
            pkl_path = os.path.join(dir_, 'case', case_id + '.pkl')
            data = misc.load_pkl(pkl_path)
            softmax = data['softmax'][:1]
            softmax = np.sort(softmax)[::-1]
            pred = softmax[:500]
          self.output_dict[mode][case_id] = pred
        except (pickle.UnpicklingError, KeyError, FileNotFoundError) as e:
          print('unable to load {} due to:'.format(pkl_path), e)
    print('loaded {} train and {} test predictions'.format(
                                              len(self.output_dict['train']), 
                                              len(self.output_dict['test'])))

  def plot_histograms(self, output='softmax'):
    # make lists of train and test values
    for mode in ['train', 'test']:
      print('\n{}'.format(mode.upper()))
      values_dict = {'softmax': {'benign': np.empty((0,2)),
                                 'tumor': np.empty((0,2))},
                     }

      for case in self.output_dict[mode].keys():
        label = misc.get_label(case) 
        values = self.output_dict[mode][case]
        n_positive = (values > 0.9).sum()
        print('{}: {:.02}'.format(case, n_positive/len(values)*100))

        values_dict['softmax'][label]= np.concatenate(
                            (values_dict['softmax'][label],
                             values))
        n, bins, patches = plt.hist(values[:,1])
        title = '{}-{}-softmax'.format(mode, case)
        plt.title(title)
        plt.ylabel('N tiles')
        plt.yscale('log')
        if output == 'softmax': 
          plt.xlim(right=1.0)
        plt.savefig(os.path.join(self.histogram_dir, title + 'jpg'))
        plt.clf()

      for category in ['benign', 'tumor']:
      # plot histograms
        n, bins, patches = plt.hist(values_dict['softmax'][category][:,1])
        title = '{}-{}-softmax'.format(mode, category)
        plt.title(title)
        plt.ylabel('N tiles')
        plt.yscale('log')
        plt.savefig(os.path.join(self.histogram_dir, title + 'jpg'))
        plt.clf()

  def pool_prediction(self, pool_type='avg', topN=0.25):
    """ Average pooling prediction using the tiles that are most likely
    to be tumor.

    # Args:
      topN: proportion of tiles to use (ie sort in descending order and 
            then use topN * num_tiles)
      output: type of output data from network to use (ie softmax or fc)
    """
    predictions_dict, labels_dict = {}, {}
    cases_dict = {}
    for mode in ['train', 'test']:
      preds, labels = np.empty(0), np.empty(0)      
      cases = []
      for case in self.output_dict[mode].keys():
        output_data = self.output_dict[mode][case]
        output_data = np.sort(output_data)[::-1]
        if pool_type == 'avg':
          n_imgs = int(np.ceil(len(output_data) * topN))
        elif pool_type == 'max':
          n_imgs = int(np.min([1, len(output_data)]))
        pred = np.mean(output_data[:n_imgs]).reshape(1)
        preds = np.concatenate((preds, pred))
        label = get_label_from_filename(case)
        label = np.array(label_to_value[label]).reshape(1)
        labels = np.concatenate((labels, label))
        cases.append(case)
      predictions_dict[mode] = preds
      labels_dict[mode] = labels
      cases_dict[mode] = cases
    return predictions_dict, labels_dict, cases_dict


# ——————————————————————————————————————————————————————————————————————

def predict_run(folders, case_split, prediction_type, 
                fv_length, test_set=None, n_process=1, 
                params={}, plot_histograms=False, save_incorrect=False,
                preprocess=False, train=False, predict=False, 
                test_subdirs=None, hyperparameter_search=False, **kwargs):
  """
  Preprocess and predict for single training run

  Args:
    folders
    case_split
    prediction_type: method of aggregating RNN region predictions
    fv_length: length of output of last layer of CNN
    n_process: number of workers for preprocessing
    threshold: probability cutoff to consider a case positive
    params: hyperparameters to tune
  """ 
  # make directory for saving incorrect images
  folders.img_save_dir = os.path.join(folders.subdir, 'incorrect_regions')
  preds, labels, cases = None, None, None
  if save_incorrect:
    misc.verify_dir_exists(folders.img_save_dir, remove_existing=True)

  if preprocess:
    p = RNNPreprocessor(folders, case_split, n_process, test_set)
    modes = list(case_split.keys())
    for mode in modes:
      p.preprocess_rnn_data(mode=mode)
  
  if train or hyperparameter_search:
    rnn_main(folders, fv_length, test_set, n_process=n_process, **params)

  if predict or hyperparameter_search:
    rnn_prediction(folders, case_split, fv_length, save_incorrect, 
                   n_process=n_process, **params) 
  
    predictor = SlidePrediction(folders, case_split)
    if plot_histograms:
      predictor.plot_histograms()
    
    prediction_methods = {
      'avg_pool': functools.partial(predictor.pool_prediction, pool_type='avg'),
      'max_pool': functools.partial(predictor.pool_prediction, pool_type='max'),
       }

    preds, labels, cases = prediction_methods[prediction_type]()
    
  return preds, labels, cases


def evaluate_run(preds, labels, threshold=0.99):
  metrics = {} 
  for mode in preds.keys():
    print(mode)
    metrics[mode] = evaluate_predictions(preds[mode], labels[mode], 
                                   threshold=threshold, print_output=True)
  return metrics['test']


# ——————————————————————————————————————————————————————————————————————

def eval_test_set(config, folders):
  """ Evalute test set by averaging predictions from models trained 
  in multi-fold cross validation runs
  """

  for i in range(1, config.n_cv+1):
    print_run_separator(i)
    folders.subdir = os.path.join(config.run_dir, 'run_{}'.format(i))    
    
    #make run folder dictionary with full paths, except for subdir names
    run_folders = get_run_folders(folders, folders.subdir)
    run_folders.test_subdirs = folders.test_subdirs

    split = {'test': config.test_list,
             'train': config.case_splits[str(i)]['train']}

    if config.predict:
      preds, labels, cases = predict_run(run_folders, split,
                                         **config)
      if isinstance(config.threshold, list):
        threshold_value = config.threshold[i-1]
      else:
        threshold_value = config.threshold

      values = preds['test'].reshape((-1, 1))
      labels = labels['test'].reshape((-1, 1)) 
      
      try:
        full_values = np.concatenate((full_values, values), 1)
      except NameError:
        full_values = values
    else: 
      predict_run(run_folders, split, **config)
  
  if config.predict:    
    avg_values = np.mean(full_values, 1)
    
    if config.print_predictions:
      output = binarize_predictions(avg_values, threshold_value)
      for i, case in enumerate(cases['test']):
        print('{}:\t{:.03} ({})\t{}'.format(case, avg_values[i], 
                                            int(output[i]), full_values[i]) )
    EvaluatePredictions(avg_values, labels, threshold_value).run()
    auc_folder = os.path.join(config.run_dir, 'auc_graphs')
    misc.verify_dir_exists(auc_folder)
    plot_roc_curve_test(avg_values, labels, folder=auc_folder, 
                        test_set=config.test_set)


def eval_cross_val_runs(config, folders):
  full_metrics = {}
  full_preds, full_labels = [], []
  cv_runs = range(1, config.n_cv+1)
  for idx in cv_runs:
    print_run_separator(idx)
    folders.subdir = os.path.join(config.run_dir, 'run_{}'.format(idx))    
    
    #make run folder dictionary with full paths, except for subdir names
    run_folders = get_run_folders(folders, folders.subdir)
    run_folders.test_subdirs = folders.test_subdirs

    split = {'test': config.case_splits[str(idx)]['test'],
             'train': config.case_splits[str(idx)]['train']}
    
    preds, labels, cases = predict_run(run_folders, split, **config)
    full_preds.append(preds['test'])
    full_labels.append(labels['test'])

    if config.print_predictions:
      print_predictions(preds['test'], labels['test'], cases['test'])
    
    if config.evaluate:
      if isinstance(config.threshold, list):
        best_threshold = determine_best_threshold(config.threshold, preds, 
                                                  labels)
      else:  
        best_threshold = config.threshold
  
  if config.evaluate:
    EvaluatePredictions(np.concatenate(full_preds),
                        np.concatenate(full_labels), config.threshold).run()
    auc_folder = os.path.join(config.run_dir, 'auc_graphs')
    misc.verify_dir_exists(auc_folder)
    plot_roc_curve_crossval(full_preds, full_labels, folder=auc_folder) 

# ——————————————————————————————————————————————————————————————————————

def rnn_hyperparameter_search(run_subdirs, subdir, case_split, threshold,
                              **kwargs):
  hyperparameters = {
    #'weight_decay': [10, 1, 0.1],
    #'ndims': [32, 64, 128],
    #'nepochs': [5],
    'weights': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
  # generate all possible combinations of hyperparameters
  iterables = tuple([hyperparameters[x] for x in hyperparameters.keys()])
  combos = list(itertools.product(*iterables))
  warnings.filterwarnings('ignore')
  for combo in combos:
    params = {k: combo[i] for i, k in enumerate(hyperparameters.keys())}
    print('\n', params)
    cv_runs = range(1, len(run_subdirs)+1)
    full_metrics = {} 
    for i in cv_runs:
      sys.stdout = open(os.devnull, 'w')
      subdir = run_subdirs[i-1]
      try:
        os.remove(os.path.join(subdir, 'rnn_checkpoint_best.pth'))
      except FileNotFoundError: 
        pass
      full_metrics[i] = evaluate_run(subdir, case_split[str(i)],
                                     threshold, params)
      sys.stdout = sys.__stdout__

    for i in cv_runs:
      run_metrics = np.array(full_metrics[str(i)]['avg_pool']
                             ).reshape((-1,1))
      try:
        metrics = np.concatenate((metrics, run_metrics), 1)
      except:
        metrics = run_metrics
      avg_metrics = [round(x, 3) for x in np.mean(metrics, 1)]
    print(avg_metrics)


# ——————————————————————————————————————————————————————————————————————

def main(FLAGS):  
  run_dir = os.path.join(FLAGS.runs_main, FLAGS.run_id)
  run_subdirs = glob.glob(os.path.join(run_dir, 'run_*'))
  
  # load slide split
  case_splits = misc.load_pkl(os.path.join(run_dir, 'case_split.pkl'))
  
  # path to csv summary file 
  handle = (FLAGS.run_id[:-2] if FLAGS.run_id.endswith('_', -3, -1)
            else FLAGS.run_id)
  csv_file = os.path.join(FLAGS.runs_main,
                          'run_summaries/{}-rnn.csv'.format(handle))

  # determine feature vector_shape
  fv_length = get_feature_vector_length(run_subdirs[0])

  # split string of thresholds
  threshold = [float(x) for x in FLAGS.threshold.split(',')]
  if len(threshold)==1:
    threshold = threshold[0]
  
  from folder_config import folder_config
  folder_config = folder_config.get(FLAGS.test_set, folder_config['train'])
  test_list = None
  if FLAGS.test_set in ['external', 'referrals']:
    test_list = misc.load_txt_file_lines(folder_config['list'])

  folders = AttributeDict({
      'region_img_dir': FLAGS.region_img_dir, 
      'test_subdirs': folder_config['image_subdirs'],
      'cnn_preds': AttributeDict({'train': 'predictions_cv/image',
                             'test': folder_config['output_dir'] + '/image'}),  
      'rnn_data': AttributeDict({'train': 'rnn_data/train',
                                 'test': 'rnn_data/' + FLAGS.test_set}),
      'rnn_preds': AttributeDict({'train': 'rnn_preds/train',
                                  'test': 'rnn_preds/' + FLAGS.test_set}),
      'rnn_case_preds': 'rnn_case_preds',

  })
  
  config = AttributeDict({
            'run_dir': run_dir,
            'run_subdirs': run_subdirs,
            'case_splits': case_splits,
            'prediction_type': FLAGS.prediction_type,
            'fv_length': fv_length,
            'n_process': FLAGS.n_process,
            'threshold': threshold,
            'csv_file': csv_file,
            'print_predictions': FLAGS.print_predictions,
            'n_cv': FLAGS.n_cv,
            'test_set': FLAGS.test_set,
            'train': FLAGS.train, 
            'predict': FLAGS.predict,
            'evaluate': FLAGS.evaluate, 
            'preprocess': FLAGS.preprocess,
            'test_list': test_list,
            'save_incorrect': FLAGS.save_incorrect,
            'hyperparameter_search': FLAGS.hyperparameter_search,
    })

  if FLAGS.hyperparameter_search:
    rnn_hyperparameter_search(run_subdirs) 
  elif FLAGS.test_set in ['external', 'referrals']:
    eval_test_set(config, folders)
  elif FLAGS.cv_idx == None:
    eval_cross_val_runs(config, folders)
  else:
     eval_single_run(FLAGS.cv_idx, config,folders)

# ——————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  ### NOTE: all file paths must be absolute ###
  parser.add_argument('--runs_main', 
      default=os.path.join(main_dir, 'results'))
  parser.add_argument('--region_img_dir',
      default=os.path.join(main_dir, 'images_5120_jpg'))
  parser.add_argument('--tile_dir',
      default=os.path.join(main_dir, 'images_40x'))
  
  parser.add_argument('--run_id')
  parser.add_argument('--n_cv', default=5, type=int)
  parser.add_argument('--cv_idx', default=None, type=int)


  parser.add_argument('--preprocess', action='store_true')
  parser.add_argument('-n', '--n_process', default=8, type=int,
    help='number of process for data preprocessing')

  parser.add_argument('--train', action='store_true')
  parser.add_argument('--predict', action='store_true')
  parser.add_argument('--evaluate', action='store_true')
  parser.add_argument('--save_incorrect', action='store_true')
  parser.add_argument('--prediction_type', default='avg_pool')
  parser.add_argument('--threshold', default='0.95',
    help='cutoff for considering a slide positive')

  parser.add_argument('--hyperparameter_search', action='store_true')
  parser.add_argument('--reset_csv', action='store_true')
  
  parser.add_argument('--test_set', default='val', 
    choices=['external', 'referrals', 'val'])
  parser.add_argument('--print_predictions', action='store_true')

  
  FLAGS = parser.parse_args()
  print('\n', FLAGS) 
  main(FLAGS)
