""" 
Script to fix prediction runs where softmax was applied on dim=0 of output 
from CNN, rather than the correct dim=1
"""

import os 
import sys 
import glob 
import pickle 
from multiprocessing import Pool

import numpy as np 
from scipy.special import softmax

import misc 

n_process = int(sys.argv[1])

def fix_file(pkl_file):
  print(pkl_file)
  data = misc.load_pkl(pkl_file)
  try:
    ndims = np.ndim(data['fc'])
    if ndims == 2:
      data['softmax'] = softmax(data['fc'], axis=1)
    elif ndims == 1:
      data['softmax'] = softmax(data['fc'], axis=0)
    else:
      print('unable to process array with {} dimensions'.format(ndims))

    misc.save_pkl(data, pkl_file)
  except KeyError as e:
    print('******** {} **********'.format(e))


def fix_folder(data_dir):
  pkl_list = glob.glob(os.path.join(data_dir, '*.pkl'))
  if n_process > 1:
    pool = Pool(n_process)
    pool.map(fix_file, pkl_list)
    pool.close(); pool.join()

  else:
    for pkl_file in pkl_list:
      fix_file(pkl_file)

if __name__ == "__main__":
  data_dirs = [
     '/projects/pathology_char/pathology_char_results/mesothelioma/results/2-23-20-resnet50/run_1/predictions_pt',
     '/projects/pathology_char/pathology_char_results/mesothelioma/results/2-23-20-resnet50/run_2/predictions_pt',
     '/projects/pathology_char/pathology_char_results/mesothelioma/results/2-23-20-resnet50/run_3/predictions_pt',
     '/projects/pathology_char/pathology_char_results/mesothelioma/results/2-23-20-resnet50/run_1/predictions_img',
     '/projects/pathology_char/pathology_char_results/mesothelioma/results/2-23-20-resnet50/run_2/predictions_img',
     '/projects/pathology_char/pathology_char_results/mesothelioma/results/2-23-20-resnet50/run_3/predictions_img',
               ]
  #data_dirs = [sys.argv[1]]

  
  for data_dir in data_dirs:
    print('\n',data_dir)
    fix_folder(data_dir)
