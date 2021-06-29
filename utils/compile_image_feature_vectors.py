import os
import sys
import argparse
import glob
import pickle
from multiprocessing import Pool
import functools
import numpy as np

run_main = '/home/alevine/mesothelioma/results/23-2-20-resnet18/'
cross_val_dirs = glob.glob(run_main + 'run_*')


if len(sys.argv) > 1:
  n_cpu = int(sys.argv[1])
else:
  n_cpu = 1

def compile_vectors(pt, img_dir, pt_dir):
  print(pt)
  fv_paths = glob.glob(os.path.join(img_dir, pt + '_*')) + \
             glob.glob(os.path.join(img_dir, pt + '-*'))
  fvs = {}
  for fv_path in fv_paths:
    with open(fv_path, 'rb') as f:
      fv = pickle.load(f)
      for k in fv.keys():
        vec = fv[k].reshape((1, -1))
        try:
          fvs[k] = np.concatenate((fvs[k], vec), 0)
        except (NameError, KeyError):
          fvs[k] = vec
  save_path = os.path.join(pt_dir, pt + '.pkl')
  with open(save_path, 'wb') as f:
    pickle.dump(fvs, f, protocol=pickle.DEFAULT_PROTOCOL)
  try:
    print(pt, fvs['softmax'].shape, fvs['fv'].shape, Flush=True)
  except KeyError as e:
    print(e, flush=True)

#def main():
       
if __name__ == '__main__':
  for subdir in cross_val_dirs[1:]:
    print(subdir)
    img_dir = os.path.join(subdir, 'predictions_img')
    pt_dir = os.path.join(subdir, 'predictions_pt')
    pts = [os.path.splitext(x)[0] for x in os.listdir(pt_dir)
           if x.startswith(('BM', 'MM'))]
    print('running on {} cases'.format(len(pts)))
    if n_cpu > 1:
      pool = Pool(n_cpu)
      pool.map(functools.partial(compile_vectors,
                                     img_dir=img_dir, pt_dir=pt_dir),
                   pts)
      pool.close(); pool.join()

    else:
      for pt in pts:
        compile_vectors(pt, img_dir, pt_dir)     
 
#  main()

