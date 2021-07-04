""" The multithreading in this script can be quite problematic for a 
few reasons. Things that have helped
- use the staintools environment with an earlier version of numpy
- try different multiprocess start methods
- adding the os.environ['OPENBLAS_NUM_THREADS'] = '1' 
- starting a new terminal window 

"""

#from multiprocessing import set_start_method; set_start_method("spawn")
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import argparse
import functools
import multiprocessing
from multiprocessing import Pool
import glob

import numpy as np
print('numpy version', np.__version__)
from PIL import Image
import staintools

root_dir = '/path/to/dir'


def process_image(image_path, normalizer, export_dir):
  #print('starting', image_path, flush=True)
  image = np.array(Image.open(image_path))
  #print('1', flush=True)
  transformed = normalizer.transform(image)
  #print('2', flush=True)
  transformed = Image.fromarray(transformed)
  #print('3', flush=True)
  save_path = os.path.join(export_dir, os.path.basename(image_path))
  #print('4', flush=True)
  transformed.save(save_path)
  print(image_path, save_path, flush=True)

def process_folder(img_dir, export_dir, normalizer, n_workers):
  if not os.path.exists(export_dir): os.makedirs(export_dir)
  image_list = glob.glob(os.path.join(img_dir, '*png'))
  print('will process {} images'.format(len(image_list)), flush=True)
  if n_workers > 1: 
    map_fn = functools.partial(process_image, 
                               normalizer=normalizer, export_dir=export_dir)
    pool = Pool(n_workers)
    pool.map(map_fn, image_list)
    pool.close(); pool.join()
  else:
    for image_path in image_list:
      process_image(image_path, normalizer, export_dir)

 

def start_process():
  print('Starting', multiprocessing.current_process().name, flush=True)


def main(FLAGS):
  target = np.array(Image.open(FLAGS.ref_image))
  normalizer = staintools.StainNormalizer(method='vahadane')
  normalizer.fit(target)
  
  # find all subdirectories within image dir or just use image dir
  if FLAGS.process_subdirs:
    subdirs = [f.name for f in os.scandir(FLAGS.image_dir) if f.is_dir() ]
    for subdir in subdirs:
      print('\n{}'.format(subdir))
      img_dir = os.path.join(FLAGS.image_dir, subdir)
      export_dir = os.path.join(FLAGS.export_dir, subdir)
      process_folder(img_dir, export_dir, normalizer, FLAGS.n_workers)

  else:
    if not os.path.exists(FLAGS.export_dir): os.makedirs(FLAGS.export_dir)
    image_list = glob.glob(os.path.join(FLAGS.image_dir, '*png'))
    print('will process {} images'.format(len(image_list)), flush=True)
    if FLAGS.n_workers > 1: 
      map_fn = functools.partial(process_image, 
                                 normalizer=normalizer, 
                                 export_dir=FLAGS.export_dir)
      pool = Pool(FLAGS.n_workers)
      pool.map(map_fn,
               image_list)
      pool.close(); pool.join()
    else:
      for image_path in image_list:
        process_image(image_path, normalizer, export_dir)


    #process_folder(FLAGS.image_dir, FLAGS.export_dir, normalizer, 
     #              FLAGS.n_workers)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref_image', default='img.png')
  parser.add_argument('--image_dir', default='dir')
  parser.add_argument('--export_dir', default='dir')
  parser.add_argument('-w', '--n_workers', default=48, type=int)
  parser.add_argument('--process_subdirs', action='store_true',
      help='look for and process by subdirectory within image_dir')
  FLAGS = parser.parse_args()

  for f in ['ref_image', 'image_dir', 'export_dir']:
    try:
      arg = vars(FLAGS)[f]
      if not os.path.isabs(arg):
        vars(FLAGS)[f] = os.path.join(root_dir, arg)
    except TypeError: pass
  
  print(FLAGS)
  # default 'fork' initializer hangs up with numpy dot function
  # see https://github.com/numpy/numpy/issues/5752
  multiprocessing.set_start_method('spawn')
  main(FLAGS)

 
