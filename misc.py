import os
import sys
import re
import pickle
import numpy as np
import openslide
import pandas as pd
from pathlib import Path
import shutil


# —————————————————————————————————————————————————————————————————————————————
# slide filemanagement

def get_tcga_id(x):
  """ Returns tcga case ID from a slide filename """
  x = os.path.basename(x)
  if len(x.split('.')) > 1:
    if len(x.split('_')) > 1:
      split = x.split('.')[0].split('_')[0].split('-')
    else:
      split = x.split('.')[0].split('-')
  else:
    split = x.split('-')
  return '{}-{}-{}'.format(split[0], split[1], split[2])


# —————————————————————————————————————————————————————————————————————————————
# model saving and loading

def load_pkl(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)

def save_pkl(obj, filename):
  with open(filename, 'wb') as f:
  	pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

# —————————————————————————————————————————————————————————————————————————————
# run logging

# option 1
class Logger(object):
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass    

def start_logger(log_file):
  if os.path.exists(log_file): os.remove(log_file)
  sys.stdout = Logger(log_file)

def end_logger():
  sys.stdout = sys.__stdout__ #.terminal
  sys.stderr = sys.__stderr__ #.terminal


# option 2; from nvidia progan script
class OutputLogger(object):
    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename, mode='wt'):
        assert self.file is None
        if os.path.exists(filename): os.remove(filename)
        self.file = open(filename, mode)
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, data):
        if self.file is not None:
            self.file.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self):
        if self.file is not None:
            self.file.flush()

class TeeOutputStream(object):
    def __init__(self, child_streams, autoflush=False):
        self.child_streams = child_streams
        self.autoflush = autoflush
 
    def write(self, data):
        for stream in self.child_streams:
            stream.write(data)
        if self.autoflush:
            self.flush()

    def flush(self):
        for stream in self.child_streams:
            stream.flush()

output_logger = None

def init_output_logging(log_file, log_stderr=True):
    global output_logger
    if output_logger is None:
        output_logger = OutputLogger()
        output_logger.set_log_file(log_file, mode='wt')
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], 
                                      autoflush=True)
        if log_stderr:
          sys.stderr = TeeOutputStream([sys.stderr, output_logger], 
                                       autoflush=True)


# —————————————————————————————————————————————————————————————————————————————
# GPU and CPU related functions

def set_gpu(gpu):
  """ sets only the specificed GPU(s) as visible
  # Args: gpu(s) as a single integer or integers separated by commas (e.g. 0,3,5)
  """
  os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
  os.environ['CUDA_VISIBLE_DEVICES']=gpu


def set_n_workers(worker_dict = {'dgx': 24, 'env': 8}):
  env = sys.executable.split('/')[-3]
  n_workers = worker_dict[env[:3]]
  print('running {} cpu threads'.format(n_workers))
  return(n_workers)



# —————————————————————————————————————————————————————————————————————————————
# image processing related functions

def evaluate_tile_quality(x, min_pixel_mean=50, max_pixel_mean=230, 
                          max_pixel_min=95):
  """ Determine if image patch meets specificed cutoffs for pixel values

  Image must be scaled from 0-255
  """
  if (x.mean() < min_pixel_mean or x.mean() > max_pixel_mean 
      or x.min() >  max_pixel_min):
    return False
  else:
    return True

def print_slide_properties(s):
  try:
    obj = s.properties['openslide.objective-power']
  except: obj = 'n/a'
  print('objective {}; MPP {:.5}; levels {}; downsamples {}'.format(
                       obj, s.properties['openslide.mpp-x'], s.level_count, 
                       [int(x) for x in s.level_downsamples]))

# —————————————————————————————————————————————————————————————————————————————

def load_labels(labels_file_path, marker, histotype, print_ns=False, **kwargs):
  """ Loads the full label file and filters for the data of interest
  # Args: 
    labels_file_path: the path to the label file
    marker: the column header for the molecular marker of interest (e.g. 'IDH.status', etc)
    histotype: either 'gbm' or 'lgg'
  # Returns: a pandas dataframe
  """
  full_df = pd.read_excel(labels_file_path)
  # select columns of interest and remove GBMs and NaNs
  df = full_df.loc[:, ['Patient.ID', 'Histology', 'Grade', marker]]
  if histotype == 'gbm':
    df = df[df.Histology == 'glioblastoma']
  elif histotype == 'lgg':
    df = df[df.Histology != 'glioblastoma']
  df = df.dropna()
  df = df[df.loc[:, marker] != 'not profiled']
  df = df.set_index('Patient.ID')
  if print_ns:
    for column in df.columns:
      print('\n{}:'.format(column)) 
      print(df[column].value_counts(dropna=False))
  return df


# —————————————————————————————————————————————————————————————————————————————

import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child
		
		from: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# —————————————————————————————————————————————————————————————————————————————
# folder management

def verify_dir_exists(x, remove_existing=False):
  if remove_existing:
    if os.path.exists(x):
      shutil.rmtree(x)
  Path(x).mkdir(parents=True, exist_ok=True)


# —————————————————————————————————————————————————————————————————————————————
# Convenience class that behaves exactly like dict(), but allows accessing 
# the keys and values using the attribute syntax, i.e., "mydict.key = value". 
 
class EasyDict(dict): 
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs) 
    def __getattr__(self, name): return self[name] 
    def __setattr__(self, name, value): self[name] = value 
    def __delattr__(self, name): del self[name] 


# —————————————————————————————————————————————————————————————————————————————

def verify_is_image(x):
  img_formats = ['jpg', 'tif', 'tiff', 'png']
  if x.split('.')[-1] in img_formats:
    return True
  else:
    return False


def get_coords(x):
  """ Determines coordinates from image filename

  Assumes that coordinates are separated from case/slide identifier
  by the first '-'. Works regardless of whether the filename has an
  extension
  """
  if os.path.isabs(x):
    x = os.path.basename(x)
  try:
    coords = x.split('-')[1].split('.')[:2]
    xint = int(coords[0])
    yint = int(coords[1])
    return xint, yint
  except IndexError: 
    print('unable to get coordinates from', x)
    return 999999, 999999

# —————————————————————————————————————————————————————————————————————————————

# run configuration management
def parse_config_file(config_file):
  with open(config_file, 'r') as f:
    lines = f.readlines()
  namespace = [x for x in lines if x.startswith("Namespace")][0]
  namespace = re.search("\(.*\)", namespace)[0][1:-1]
  items = namespace.split(', ')
  config = {}
  for item in items:
    k, v = item.split('=')
    v = v.strip("'")
    if v.isdigit():
      v = int(v)
    config[k] = v
  return config


# —————————————————————————————————————————————————————————————————————————————

def load_txt_file_lines(x):
  list_ = []
  with open(x, 'r') as f:
    for line in f:
      list_.append(line.rstrip())
  return list_



