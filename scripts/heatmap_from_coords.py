import os
import sys
import glob
import time
import argparse
from multiprocessing import Pool
import functools

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
import matplotlib.pyplot as plt
import openslide
import skimage as ski
from skimage import transform
import cv2

import misc
from slide_prediction import get_label_from_filename

slides_main = '/projects/pathology_char/pathology_char_results/mesothelioma/slides/'
slide_dirs = {
  'tumor': os.path.join(slides_main, 'malig_meso_sarc'),
  'benign': os.path.join(slides_main, 'benign_meso_spin'),
}

TILE_DIM = 512


def make_heatmap(slide, tile_list, data, save_dir, rescale=True, 
                 save_format='jpg'):
  slide_dir = slide_dirs[get_label_from_filename(slide)]
  slide_path = os.path.join(slide_dir, slide + '.svs')
  try:
    slide_obj = openslide.OpenSlide(slide_path)
  except openslide.lowlevel.OpenSlideUnsupportedFormatError:
    print('unable to load:', slide_path)
    return
  width, height = slide_obj.dimensions
  ydim = int(round(height/TILE_DIM))
  xdim = int(round(width/TILE_DIM))
  pred_arr = np.zeros((xdim, ydim))
  for tile in tile_list:
    x, y = misc.get_coords(tile)
    x = int(np.floor(x/TILE_DIM))
    y = int(np.floor(y/TILE_DIM))
    pred = data[tile]['softmax'][1]
    pred_arr[x, y] = pred
  # set size of final image
  slide_img = np.array(slide_obj.get_thumbnail((xdim*8, ydim*8)))
  preds = np.transpose(pred_arr)
  #import pdb;pdb.set_trace()
  if rescale:
    # rescale so that 0.9 and below=0 and then normalize
    preds[preds < 0.9] = 0.9
    preds = (preds-0.9)/(1.0-0.9)
  preds = (preds *255).astype('uint8')
  heatmap = cv2.applyColorMap(preds, cv2.COLORMAP_VIRIDIS)
  #heatmap = (heatmap *255).astype('uint8')
  dim1, dim2 = slide_img.shape[0], slide_img.shape[1]
  heatmap = cv2.resize(heatmap, (dim2, dim1))
  fin = cv2.addWeighted(slide_img, 0.6, heatmap, 0.4, 0)
  save_path = os.path.join(save_dir, slide + '.' + save_format)
  cv2.imwrite(save_path, fin)
  print(slide, flush=True)


def process_case(pkl, heatmap_dir):
  data = misc.load_pkl(pkl)
  # pull out distinct slides
  slide_list = list(set([os.path.basename(x).split('-')[0] 
                         for x in data.keys()]))
  for slide in slide_list:
    # pull tiles corresponding to each slide
    tile_list = [x for x in data.keys() if x.startswith(slide)]
    # make heatmap
    make_heatmap(slide, tile_list, data, save_dir=heatmap_dir)

def process_run(subdir, case_split, heatmap_main, FLAGS):
  cv_idx = subdir[-1]
  pred_dir = os.path.join(subdir, 'predictions_cv/image')
  heatmap_dir = os.path.join(heatmap_main, 'run_{}'.format(cv_idx))
  misc.verify_dir_exists(heatmap_dir, remove_existing=FLAGS.remove_existing)
  pkl_list = glob.glob(os.path.join(pred_dir, '*pkl'))
  test_list = case_split[cv_idx]['test']
  pkl_list = [x for x in pkl_list 
              if os.path.basename(x).split('.')[0] in test_list]
  if FLAGS.n_process > 1:
    pool = Pool(FLAGS.n_process)
    pool.map(functools.partial(process_case, heatmap_dir=heatmap_dir),
             pkl_list)
    pool.close(); pool.join() 
    
  else:
    for pkl in pkl_list:
      process_case(pkl, heatmap_dir)


def main(FLAGS):
  case_split = misc.load_pkl(os.path.join(runs_main, FLAGS.run_id, 
                                          'case_split.pkl'))
  heatmap_main = os.path.join(runs_main, FLAGS.run_id, 'heatmaps')
  print('transfer: scp -r alevine@xfer.bcgsc.ca:{} .'.format(heatmap_main))
  misc.verify_dir_exists(heatmap_main)
  if FLAGS.cv_run:
    subdir = os.path.join(runs_main, FLAGS.run_id, 
                          'run_{}'.format(FLAGS.cv_run))
    process_run(subdir, case_split, heatmap_main, FLAGS)
  else:
    for subdir in glob.glob(os.path.join(runs_main, FLAGS.run_id, 'run_*')):
      process_run(subdir, case_split, heatmap_main, FLAGS)


if __name__ == "__main__":
  runs_main = '/home/alevine/mesothelioma/results'
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_id', default='10-23-20-normjit',)
  parser.add_argument('--cv_run', default=None, type=int)
  parser.add_argument('--n_process', default=1, type=int)
  parser.add_argument('--remove_existing', action='store_true')
  FLAGS = parser.parse_args()
  main(FLAGS)

