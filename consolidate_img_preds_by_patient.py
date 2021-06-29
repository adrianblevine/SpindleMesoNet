import os
import sys
import argparse
import glob
import pickle
from multiprocessing.dummy import Pool
import functools
import numpy as np

results_main = '/home/alevine/mesothelioma/results'


def compile_vectors(pt, old_dir, new_dir, img_pred_list):
  img_preds = [x for x in img_pred_list if pt in x] 

  pt_dict = {}
  for img in img_preds:
    with open(os.path.join(old_dir, img), 'rb') as f:
      data = pickle.load(f)
    pt_dict[img] = data

  save_path = os.path.join(new_dir, pt + '.pkl')
  with open(save_path, 'wb') as f:
    pickle.dump(pt_dict, f, protocol=pickle.DEFAULT_PROTOCOL)

  print('{}: {}'.format(pt, len(pt_dict.keys())), flush=True)

def determine_if_process_run(run):
  old_dir = os.path.join(results_main, run, 'run_3/predictions_img')
  new_dir = os.path.join(results_main, run, 'run_3/predictions_img_2')
  if os.path.isdir(new_dir):
    if len(os.listdir(new_dir)) > 145:
      return False
  if os.path.isdir(old_dir):
    return True
  return False


if __name__ == '__main__':

  # make list of all runs with predictions
  runs = [x for x in os.listdir(results_main) 
          if determine_if_process_run(x)==True]
  runs.reverse()
  print(runs)

  # determine number of cpus
  if len(sys.argv) > 1:
    n_cpu = int(sys.argv[1])
  else:
    n_cpu = 1
  
  for run in runs:
    print('\n{}'.format(run.upper()))
    # loop over subdirs
    cross_val_dirs = glob.glob(os.path.join(results_main, run, 'run_*'))

    for subdir in cross_val_dirs:
      print('\n{}'.format(subdir))
      # make lists of images and cases
      old_dir = os.path.join(subdir, 'predictions_img')
      new_dir = os.path.join(subdir, 'predictions_img_2')
      if not os.path.isdir(new_dir): os.makedirs(new_dir)
      img_pred_list = os.listdir(old_dir)
      pts = [os.path.splitext(x)[0] 
             for x in os.listdir(os.path.join(subdir, 'predictions_pt'))
             if x.startswith(('BM', 'MM'))]
      print('running on {} cases'.format(len(pts)))

      #if n_cpu > 1:
      pool = Pool(48)
      pool.map(functools.partial(compile_vectors,
                                   old_dir=old_dir, new_dir=new_dir,
                                   img_pred_list=img_pred_list),
                     pts)
      pool.close(); pool.join()

      #else:
      #  for pt in pts:
      #    compile_vectors(pt, old_dir, new_dir, img_pred_list)     
 

