
""" Makes heatmaps of 2017 slides from a trained model

# note: when working with openslide, remember that openslide lists its
    dimensions as (horizontal axis, vertical axis), while numpy does the
    opposite (ie the 0 and 1 axes are flipped in openslide)
    the coordinates returned by tissue detection are in the 
    format for PIL/openslide (col, row), and need to be reversed for use by 
    numpy/matplotlib (row, col) - however both have origin in top left

"""

import os
import sys
import glob
import time
import argparse

import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import openslide
import skimage as ski

import torch
from torch import nn
from torchvision import transforms

from tissue_detection import TissueDetection
import misc
from pth_models import initialize_model
import main_run

PATHS = misc.EasyDict({
  'root_dir': '/projects/pathology_char/pathology_char_results/tcga_glioma/molecular',
  'slide_dir': '/projects/ovcare/tcga_glioma/slides/',
  'labels_file': '/projects/pathology_char/pathology_char_results/tcga_glioma/gbmlgg_051419.xlsx',
   })

FLAGS = []



class Heatmap(TissueDetection):
  """ Loads a slide and generates the heatmap 
  """
  def __init__(self, slide_path, model, device, *args, **kwargs):
    #super().__init__(slide_path)
    self.slide_id = os.path.basename(slide_path)
    self.slide_obj = openslide.OpenSlide(slide_path)
    self.device = device
    self.model = model.to(self.device)
    self.output_dir = kwargs['output_dir']
    self.generate_mask()
    #print([x for x in dir(self) if x[:2] != '__'])
   
  
  def heatmap_by_batch(self, side=1024,  batch_size=12, 
                       debug=False, min_pixel_mean=50, max_pixel_mean=250, 
                       max_pixel_min=100, tissue_proportion_cutoff=0.95):
    """ Generates the heatmap
    
    # Arguments
        side
        div: denotes the downscaling of the FCN output vs its 
            input; this will depend on how the model was trained 
        batch_size
        debug: prints significantly more output
        min_pixel_mean: mean pixel value below which a tile will be discarded
        max_mixel_mean: mean pixel value above which a tile will be discarded
        max_pixel_min: tile will be discarded if its minimum pixel value 
            is above this
        tissue_proportion_cutoff: minimum amount of tile that is within 
            tissue mask region, or else tile will be discarded
    """
    self.heatmap = np.zeros((self.slide_obj.dimensions[0]//side, 
                             self.slide_obj.dimensions[1]//side), dtype=np.float32)
    #print('tissue mask shape:', tissue_mask.size)
    xmin, xmax, ymin, ymax = self.bounding_box  
    print('slide dimensions: {}; bounding box {}'.format(self.slide_obj.dimensions, 
                                                         self.bounding_box)) 
    currentx, currenty = xmin, ymin
    batch_index = 0
    total_batch = 0
    tiles = torch.empty((0,3,side,side), dtype=torch.float32)
    coords = np.empty((0,2))
    # torchvision models require images in format [c,h,w], scaled from 0-1,
    # and normalized as below, see https://pytorch.org/docs/stable/torchvision/models.html
    img_transforms = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
                     ])
    while currenty <= ymax:
      while currentx <= xmax:
        while tiles.shape[0] < batch_size:
          d = int(round(self.slide_obj.dimensions[0]/self.tissue_mask.size[0]))
          tissue_tile = np.array(self.tissue_mask.crop((currentx/d, currenty/d,
                                                        (currentx+side)/d, (currenty+side)/d)))
          if tissue_tile.mean() > tissue_proportion_cutoff:
            try:
              tile_raw = self.slide_obj.read_region((currentx,currenty), 0, (side,side))
              tile_raw = np.array(tile_raw)[:,:,:3].astype(np.uint8)
            except openslide.lowlevel.OpenSlideError:
              print('OpenSlideError at coords', currentx, currenty)
              tile_raw = np.zeros((side, side, 3), dtype=np.uint8)
            if misc.evaluate_tile_quality(tile_raw):
              # very important to make sure the preprocessing function is 
              # working properly or else heatmaps will all be wrong
              tile_norm = img_transforms(tile_raw).reshape((1, 3, side, side))
              assert tile_norm.max() < 5 # ensure image patch is properly scaled
              tiles = torch.cat((tiles,tile_norm), dim=0)
              # NOTE: coords either need to be divided here or when 
              # the prediction tile is pasted into the final heatmap; always 
              # remember that x and y need to be flipped for PIL vs numpy
              coord = np.array([currentx//side, currenty//side])
              coord = np.reshape(coord, (1,2))
              coords = np.append(coords, coord, axis=0)
          if currentx <= xmax:
            currentx += side
          elif currentx > xmax and currenty <= ymax:
            currentx = xmin
            currenty += side
            if debug: print('y:', currenty, 'row:', (currenty-ymin)//side)
            if not int((currenty-ymin)//side) % 100:
              print('  row %d/%d' % ((currenty-ymin)//side, (ymax-ymin)//side))
          elif currenty > ymax:
            print('ymax reached at coordinate {0}, row {1}'.format(currenty, 
                                                       (currenty-ymin)//side))
            break
        if tiles.shape[0] > 0:
          total_batch +=1
          if not total_batch % 200 and debug: 
            print('  batch', total_batch)
          # run pytorch model on input tiles
          tiles = tiles.to(self.device)
          predictions = self.model(tiles)
          for i in range(len(predictions)):
            if debug and i==0: 
              print(predictions[i,0,:5,1])
            pixel_pred = predictions[i,1].item()
            self.heatmap[int(coords[i,0]), int(coords[i,1])] = pixel_pred
          tiles = torch.empty((0,3,side,side), dtype=torch.float32)
          coords = np.empty((0,2))
        if currenty > ymax:    
          if debug:
            print('ymax reached at row', (currenty-ymin)//side)
          break
      currenty += side
      currentx = xmin
      if debug: 
        print(currenty, (currenty-ymin)//side)
    print('total tiles predicted: {}; mean {}, min {}, max {}'.format(total_batch*batch_size,
          self.heatmap.mean(), self.heatmap.min(), self.heatmap.max() ) )
    try:
      np.save(os.path.join(self.output_dir, self.slide_id[:-3] + 'npy'), 
              self.heatmap.astype('float16'))
      hmap_im = Image.fromarray((self.heatmap*255).astype('uint8'), mode='L')
      hmap_im.save(os.path.join(self.output_dir, self.slide_id[:-3] + 'png'))
    except (KeyError, OSError) as e:
      print('Unable to save image:', e)


  def plot_mask(self):
    plt.imshow(self.heatmap)
    plt.show()


  def plot_prediction(self, size = 30):
    slide_im = np.array(self.slide_obj.get_thumbnail((2000,2000)))
    heatmap = ski.transform.resize(np.array(self.heatmap), slide_im.shape[:2])
    ratio = self.slide_obj.dimensions[1]/self.slide_obj.dimensions[0]
    fig = plt.figure(figsize = (size, size*ratio))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    plt.imshow(slide_im, interpolation='none')
    plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.15)
    plt.show()

# ———————————————————————————————————————————————————————————————————————

def load_model_with_softmax(model_type, run_dir, epoch_weights=None):
  """ Loads a model and weights, adding a final softmax layer

  # Arguments
  
  # Returns: a pytorch model
  """
  ckpt_path = find_ckpt_path(run_dir, epoch_weights)
  base_model = initialize_model(model_type)
  base_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
  model = nn.Sequential(base_model,
                        nn.Softmax() 
                        )
  model.eval()
  return model

  
def find_ckpt_path(run_dir, epoch_weights=None):
  if epoch_weights is not None:
    ckpt_path = glob.glob(os.path.join(run_dir, 'ckpt*{}*pth'.format(epoch_weights)))[0]
  elif os.path.exists(os.path.join(run_dir, 'ckpt_best_val.pth')):
    ckpt_path = os.path.join(run_dir, 'ckpt_best_val.pth')
  else:
    ckpt_path = sorted(glob.glob(os.path.join(run_dir, 'ckpt*pth')),
                       key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[2]))[-1]
  print('loading weights from:', ckpt_path)
  return ckpt_path


def make_slide_list(slide_split, phase='test'):
  """ Makes a list of slides to process according to command line args
  """
  if 'test' in slide_split.keys():
    slide_list = slide_split[phase]
    slide_list = [os.path.join(PATHS.slide_dir, x) for x in slide_list]
    slide_list.sort()
   
    if not FLAGS.overwrite:
      mask_list = [x.split('.')[0] for x in os.listdir(PATHS.output_dir)]
      slide_list = [x for x in slide_list if os.path.basename(x).split('.')[0] 
                    not in mask_list] 
    print('number of slides to process:', len(slide_list))
  else: pass
  
  list_slide_dir = os.path.dirname(slide_list[0])
  if list_slide_dir != PATHS.slide_dir:
    print('slide directory in slide list is {}, while specificed slide directory is is {}'
          '\nwill correct all paths in slide list'.format(list_slide_dir, PATHS.slide_dir))
    slide_list = [os.path.join(PATHS.slide_dir, os.path.basename(x)) for x in slide_list]

  return slide_list


def process_single_slide(slide_id, plot_heatmap=False):
  slide_path = os.path.join(PATHS.slide_dir, slide_id)
  print(slide_path)
  slide = Heatmap(slide_path, model, device=device, output_dir=output_dir)
  slide.heatmap_by_batch(side=FLAGS.dimension, debug=FLAGS.debug_mode)
  slide.plot_prediction()


def process_slide_list(slide_list, heatmap_args, labels_df, **kwargs):
  time_list = []
  print('will generate heatmaps for {} slides'.format(len(slide_list)))
  for slide_path in slide_list:
    start_time = time.time()
    try:
      tcga_id = misc.get_tcga_id(os.path.basename(slide_path))
      column = main_run.marker_columns[FLAGS.marker]
      print('\n', tcga_id, labels_df[column][tcga_id])
      slide = Heatmap(slide_path, **heatmap_args)
      slide.heatmap_by_batch(side=FLAGS.dimension)
    except (openslide.lowlevel.OpenSlideError) as e:
      print('unable to process slide, due to error:', e)
      print(slide_path)
    print('time: {:.1f} min'.format(((time.time()-start_time)/60))) 
    time_list.append(time.time()-start_time)
  print('\nmean time taken: {:.1f} min'.format(np.array(time_list).mean()/60))  


def load_run_config(logfile):
  with open(logfile, 'r') as f:
    lines = f.read()
  pat = re.search('Namespace(.*)', lines)
  x = pat.group(1)[1:-1]
  x = x.split(',')
  x = [i.replace(' ', '').split('=') for i in x]

  def strip_quotes(x):
    if x[0] == "'": 
      x = eval(x)
    return x

  config = {i[0]: strip_quotes(i[1]) for i in x}
  for k in config.keys():
    try:
      config[k] = int(config[k])
    except ValueError: pass
  return config


def _main(FLAGS):
  misc.set_gpu(FLAGS.gpu)
  PATHS.run_dir = os.path.join(PATHS.root_dir, 'runs-' + FLAGS.marker, FLAGS.run_id)
  PATHS.output_dir = os.path.join(PATHS.run_dir, 'heatmaps/')  
  misc.verify_dir_exists(PATHS.output_dir)
  misc.init_output_logging(os.path.join(PATHS.output_dir, 'logfile.txt'))

  assert torch.cuda.is_available()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  labels_df = misc.load_labels(PATHS.labels_file, main_run.marker_columns[FLAGS.marker], 
                               FLAGS.histotype)

  if FLAGS.slide_list:
    slide_list = FLAGS.slide_list.split(',')
    model = load_model_with_softmax(FLAGS.model_type, PATHS.run_dir, FLAGS.epoch_weights)
    print('will process the following slides:', slide_list)
    for slide_id in slide_list:
      process_single_slide(slide_id)

  else:
    slide_split = misc.load_pkl(os.path.join(PATHS.run_dir, 'slide_split.pkl'))
    if 'test' in slide_split.keys():
      slide_list = make_slide_list(slide_split)
      model = load_model_with_softmax(FLAGS.model_type, PATHS.run_dir, FLAGS.epoch_weights)
      heatmap_args = {'model': model, 'device': device, 'output_dir': PATHS.output_dir}
      process_slide_list(slide_list, heatmap_args, labels_df)
    else:
      if FLAGS.epoch_weights is not None:
        epoch_weight_list = FLAGS.epoch_weights.split(',')
      for run_idx in slide_split.keys():
        print('\n\nRUN {}:'.format(run_idx))
        try: 
          epoch_weights = epoch_weight_list[int(run_idx)-1]
        except NameError: 
          epoch_weights=None
        run_subdir = os.path.join(PATHS.run_dir, 'run_{}'.format(run_idx))
        output_subdir = os.path.join(PATHS.output_dir, 'run_{}'.format(run_idx))
        misc.verify_dir_exists(output_subdir)
        slide_list = make_slide_list(slide_split[run_idx])
        model = load_model_with_softmax(FLAGS.model_type, run_subdir, epoch_weights)
        heatmap_args = {'model': model, 'device': device, 'output_dir': output_subdir}
        process_slide_list(slide_list, heatmap_args, labels_df)

# ———————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', required=True, 
      help="GPU to use")
  parser.add_argument('--dimension', type=int, default=1024,
      help='Side dimension for tiles')
  parser.add_argument('--marker', choices=main_run.marker_columns.keys() )
  parser.add_argument('--histotype', choices=['lgg', 'gbm'])
  parser.add_argument('--run_id', default=None,
      help="Saved model to use for generating predictions")
  parser.add_argument('--phase', choices=['test', 'val'])
  parser.add_argument('--epoch_weights', default=None, type=int,
      help='Train epoch from which saved weights should be loaded.')
  parser.add_argument('--model_type', default='resnet18')

  parser.add_argument('--slide_list', required=False,
      help='Specify a single slide or list of slides to evaluate. Separate each with '
           'a comma but no spaces. Must be in the same folder')
  parser.add_argument('--debug_mode', action='store_true',
      help='Generates significantly more output to help with debugging')
  parser.add_argument('--overwrite', action='store_true',
      help='Overwrite all saved files when evaluating a folder')
  parser.add_argument('--batch_size', default=12, type=int,
      help='The default batch size is optimized for running with one gpu on the devbox')
  FLAGS = parser.parse_args()
  
  if FLAGS.run_id is None:
    runs = os.listdir(os.path.join(PATHS.root_dir, 'runs-' + FLAGS.marker))
    runs.sort(key=lambda x: list(reversed(x.split('-'))))
    FLAGS.run_id = runs[-1] 

  print(FLAGS)
  PATHS.slide_dir = os.path.join(PATHS.slide_dir, 'tcga_' + FLAGS.histotype) 
  _main(FLAGS)
