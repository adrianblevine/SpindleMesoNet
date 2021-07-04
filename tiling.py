#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Load slide annotations, generate annotation mask, and export image patches
"""

import sys
import glob
import os
import argparse
import time
import random
import functools
from multiprocessing import Pool
import re
import ctypes
import pickle
import datetime
from distutils import util
import json

import openslide as openslide
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tissue_detection import TissueDetection
import misc

FLAGS = []
ROOT_DIR = '/projects/pathology_char/pathology_char_results'

# list all possible categories in annotations, while export_categories
# only includes image labels to be saved
annotation_categories = ['tumor', 'blood', 'stroma', 'necrosis', 'other']

print = functools.partial(print, flush=True)

def get_tcga_id(x):
  """ Returns tcga patient ID from a slide filename """
  split = x.split('.')[0].split('-')
  return '{}-{}-{}'.format(split[0], split[1], split[2])


class SlideTiling(TissueDetection):
  def __init__(self, slide_path, index, export_dir, mask_dir, 
               annotation_dir=None, resolution=40,):
    super().__init__(slide_path)
    self.slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    self.slide_path = slide_path
    self.export_dir = export_dir
    self.mask_dir = mask_dir
    self.slide_obj = openslide.OpenSlide(self.slide_path)
    self.index = index

    self.resolution = resolution
    self.objective_power = int(self.slide_obj.properties[
                               'openslide.objective-power'])
    self.level_to_objective = {
      i: round(self.objective_power/self.slide_obj.level_downsamples[i], 2)
      for i in range(self.slide_obj.level_count)}
    self.objective_to_level = {v: k for k, v in self.level_to_objective.items()}
    try:
      self.level = self.objective_to_level[self.resolution]
      self.sample_factor = 1
      self.multiple = round(self.slide_obj.level_downsamples[self.level])
    except KeyError:
      self.level = 0
      self.sample_factor = round(self.level_to_objective[0]/self.resolution)
      self.multiple = 1
      print(self.slide_id, 'no level for {}x resolution, will crop from ' 
            'level {} and downsample by {}'.format(self.resolution, 
                                                   self.level, 
                                                   self.sample_factor),
            flush=True)
    self.slide_dims = self.slide_obj.level_dimensions[self.level]
    
    try:
      self.tissue_mask = Image.open(os.path.join(self.mask_dir,
                                                 self.slide_id + '.tif'))
    except FileNotFoundError:
      print('unable to find saved mask for {}'.format(self.slide_id))
      self.generate_mask(downsample=16)
    self.tissue_area = np.array(self.tissue_mask).sum()
    
    if annotation_dir is not None:
      self.annotation_categories = annotation_categories
      self.annotation_path = glob.glob(os.path.join(annotation_dir, 
                                       self.slide_id + '*'))[0]
      self.generate_annotation_mask()
      # needs to be defined in order to set stride in export args
      self.tissue_area = 0 
      

  def export_tiles_grid(self, patch_size=1024, stride=1., 
                        tissue_proportion_cutoff=0.95, img_format='png', 
                        max_exports=10000, dummy_round=False, 
                        use_coordinates=False, **kwargs):
    """ Tiles the slide across all rows and columns in a grid
    
    # Arguments
        patch_size: dimension of tiles; output shape will be 
                    (patch_size, patch_size, 3)
        tissue_proportion_cutoff: minimum proportion of tissue pixels for export
        img_format
        max_exports: maximum number of images exported per slide
    """
    start_time = time.time()
    crop_size = round(patch_size * self.sample_factor)
    step_size = round(crop_size * stride)
    div = round(self.slide_dims[0]/self.tissue_mask.size[0])
    xmax, ymax = self.slide_dims[0:2]
    crops, exports, excluded = 0, 0, 0
    x, y = 0, 0
    while y < (ymax-crop_size) and exports < max_exports:
      while x < (xmax-crop_size) and exports < max_exports:

        mask_tile = self.tissue_mask.crop((x//div, y//div, 
                                          (x+crop_size)//div, 
                                          (y+crop_size)//div))
        mask_tile = np.array(mask_tile).astype('uint8')
        tissue_proportion = (np.count_nonzero(mask_tile)/mask_tile.size)
        crops += 1
        if tissue_proportion > tissue_proportion_cutoff:
          if use_coordinates == True:
            # x and y divided by sample factor to maintain coordinates 
            # consistent with the resolution
            file_name = '{0}-{1}.{2}.{3}'.format(self.slide_id, 
                                                 round(x/self.sample_factor), 
                                                 round(y/self.sample_factor), 
                                                 img_format)
          else:
            file_name = '{0}-{1}.{2}'.format(self.slide_id, exports, img_format)
          
          # location (tuple) – (x, y) tuple giving the top left pixel 
          # in the *level 0 reference frame*
          # returns RGBA image, which is converted to RGB
          tile = self.slide_obj.read_region(  
                                          (x*self.multiple, y*self.multiple),
                                          self.level, 
                                          (crop_size, crop_size)).convert('RGB')
          if self.sample_factor != 1:
            # Lanczos resample: "a high-quality downsampling filter"
            tile = tile.resize((patch_size, patch_size), 
                               resample=Image.LANCZOS)
          
          if misc.evaluate_tile_quality(np.array(tile)):
            if not dummy_round:
              tile.save(os.path.join(self.export_dir, file_name))
            exports += 1
          else: 
            excluded += 1
            #if excluded % 100 == 0:
            #  tile.save(os.path.join(self.export_dir, 
            #                         'EXCL-' + file_name[:-3] + 'jpg'))
        x += step_size
      x = 0
      y += step_size
    print('\n{}: {}  {}'.format(self.index, self.slide_id, 
                                self.slide_obj.dimensions))
    print('  tissue area: {}'.format(np.array(self.tissue_mask).sum()))
    print('  time taken: {:0.2f} min'.format((time.time()-start_time)/60))
    print('  exports/crops/excluded: {}/{}/{} (stride={})'.format(exports, 
          crops, excluded, stride))
    return exports


  def export_tiles_with_annotations(self, patch_size=1024, xstride=1., 
                                    ystride=1., annot_proportion_cutoff=0.95, 
                                    tissue_proportion_cutoff=0.95,
                                    img_format='png', max_exports=99999, 
                                    export_categories=None, 
                                    dummy_round=False, use_coordinates=False,
                                    **kwargs):
    start_time = time.time()
    crop_size = round(patch_size * self.sample_factor)
    xstep_size = round(crop_size * xstride)
    ystep_size = round(crop_size * ystride)    
    value_to_category = {v: k for k,v in self.category_values.items()}
    
    div = round(self.slide_dims[0]/self.tissue_mask.size[0])
    annotation_mask = self.annotation_mask.resize(self.tissue_mask.size)
    
    xmax, ymax = self.slide_dims[0:2]
    xmin, ymin = 0, 0
    
    img_exports = {x: 0 for x in export_categories}
    exports, crops, excluded = 0, 0, 0 
    x, y = xmin, ymin
    
    while y < (ymax - crop_size) and exports < max_exports:
      while x < (xmax - crop_size) and exports < max_exports:
        annot_tile = annotation_mask.crop((x//div, y//div, 
                                          (x+crop_size)//div, 
                                          (y+crop_size)//div))
        annot_tile = np.array(annot_tile).astype('uint8')
        annot_proportion = (np.count_nonzero(annot_tile)/annot_tile.size)
        if annot_proportion > annot_proportion_cutoff:
          try:
            # get most frequent value above 0 and corresponding category
            top_value = np.bincount(annot_tile[annot_tile > 0].ravel()
                                    ).argmax() 
            category = value_to_category[top_value]
          except (KeyError, TypeError) as e:
            print(e, '- unable to match category for:', annot_tile.mean())
            pass

          # determine proportion within tissue mask
          mask_tile = self.tissue_mask.crop((x//div, y//div, 
                                          (x+crop_size)//div, 
                                          (y+crop_size)//div))
          mask_tile = np.array(mask_tile).astype('uint8')
          tissue_proportion = (np.count_nonzero(mask_tile)/mask_tile.size)
          if (category in export_categories 
              and tissue_proportion > tissue_proportion_cutoff):
            try:
              tile = self.slide_obj.read_region(
                                    (x*self.multiple, y*self.multiple), 
                                    self.level,
                                    (crop_size,crop_size)).convert('RGB')
            except ctypes.ArgumentError as e:
              print('unable to crop tile due to {}'.format(e))
            crops += 1
            if self.sample_factor != 1:
              tile = tile.resize((patch_size, patch_size), 
                                 resample=Image.LANCZOS)
            
            if use_coordinates:
              tile_id = '{}.{}'.format(round(x/self.sample_factor), 
                                       round(y/self.sample_factor))
            else:
              tile_id = img_exports[category]

            if len(export_categories) > 1:
              file_name = '{}-{}-{}.{}'.format(self.slide_id, category,   
                                               tile_id, img_format)
            else: 
              file_name = '{}-{}.{}'.format(self.slide_id, 
                                            tile_id, img_format) 
            
            if misc.evaluate_tile_quality(np.array(tile)):
              if not dummy_round:
                tile.save(os.path.join(self.export_dir, file_name))
              img_exports[category] += 1
            else: 
              excluded += 1
              #if excluded % 100 == 0:
              #  tile.save(os.path.join(self.export_dir, 
              #                         'EXCL-' + file_name[:-3] + 'jpg'))
        x += xstep_size
      x = xmin
      y += ystep_size
    exports = sum(img_exports.values())
    print('\n{}: {}  {}'.format(self.index, self.slide_id, 
                                self.slide_obj.dimensions))
    print('  time taken: {:0.2f} min'.format((time.time()-start_time)/60))
    print('  exports/crops/excluded: {}/{}/{} (strides = {}, {})'.format(
          exports, crops, excluded, xstride, ystride))
    return exports


  def generate_annotation_mask(self):
    self.annotations = {}
    x_values = []
    y_values = []
    for category in self.annotation_categories:
      self.annotations[category] = []
    with open(self.annotation_path, 'r') as f:
      # for original annotation export in qupath 0.1
      if self.annotation_path.endswith('txt'):
        for line in f:
        # need to set the dividing character between category and coordinates
          category = line.split(' ')[0].lower()
          if 'Point' in line:
            xy= [float(s) for s in re.findall(r'-?\d+\.?\d*', line)]
            x = xy[0::2]
            y = xy[1::2]
            x_values += x
            y_values += y
            coords = []
            for i in range(len(x)):
              coords.append((x[i], y[i]))
            #try:
            self.annotations[category].append(coords)
            #except KeyError:
            #  self.annotations[category] = []
            #  self.annotations[category].append(coords)
          else:
            pass
      # json annotation export with qupath 0.2
      elif self.annotation_path.endswith('json'):
        json_ = json.load(f)
        for item in json_:
          try:
            category = item['properties']['classification']['name'].lower()
            coords_list = item['geometry']['coordinates']
          except KeyError:
            category = 'tumor'
            coords_list = item['coordinates']
          if len(coords_list) > 1:
            coords_list = [x[0] for x in coords_list]
          for coords in coords_list:
            x = [i[0] for i in coords]
            y = [i[1] for i in coords]
            x_values += x
            y_values += y
            polygon = []
            for i in range(len(x)):
              polygon.append((x[i], y[i]))
            self.annotations[category].append(polygon)
    try:
      self.bounding_box = (int(min(x_values)), int(max(x_values)), 
                         int(min(y_values)), int(max(y_values)))
    except TypeError:
      self.bounding_box = (0, 999999, 0, 999999)

    # need to set pixel values for each category
    self.category_values = {self.annotation_categories[i]: i + 1 
                              for i in range(len(self.annotation_categories))}
    height, width = self.slide_obj.dimensions[0:2]
    self.annotation_mask = Image.new('L', (height, width), 0)
    annotation_labels = (annotation_categories + 
                             [x for x in self.annotations.keys()
                             if x not in annotation_categories])
    # make sure tumor is the first item in list so that any holes
    # in tumor region (e.g. necrosis) are drawn after tumor
    if 'tumor' in annotation_labels:
      annotation_labels.remove('tumor')
      annotation_labels.insert(0, 'tumor')
    
    # draw annotations as polygons
    for category in annotation_labels:
      for coordinates in self.annotations[category]:
        try:
          ImageDraw.Draw(self.annotation_mask).polygon(coordinates,
                                 fill=self.category_values.get(category, 0),
                                 outline=self.category_values.get(category, 0))
        except TypeError as e:
          print('{}: unable to generate {} annotation due to {}'.format(
                self.slide_id, category, e))

  def view_mask_overlaid_on_slide(self, mask_save_dir=None, size=8, 
                                  save_only=True):
    """ plots or saves the annotation mask overlaid on the slide """
    tissue = self.slide_obj.get_thumbnail((5000,5000))
    try:
      mask = self.annotation_mask.resize(tissue.size)
    except:
      mask = self.tissue_mask.resize(tissue.size)
    if tissue.size[1] > tissue.size[0]:
      tissue = tissue.rotate(90, expand=True)
      mask = mask.rotate(90, expand=True)
    tissue = np.asarray(tissue)
    mask = np.asarray(mask)/3
    #ratio = tissue.shape[0]/tissue.shape[1]
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    plt.imshow(tissue, interpolation='none')
    plt.imshow(mask, 'jet', interpolation='none', alpha=0.3)
    fig.tight_layout()
    if save_only:
      plt.savefig(os.path.join(mask_save_dir, self.slide_id + '.jpg'),
                  format='jpg', orientation='landscape')
      plt.close()
    else:
      plt.show()

  def export_tiles_random(self, patch_size=1024, tissue_proportion_cutoff=0.99, 
                          img_format='png', max_exports=100, max_crops=1000, 
                          **kwargs):
    """ Tiles the slide by randomly selecting patches to crop
    
    *** CURRENTLY NOT WORKING ***
    
    # Arguments
        patch_size: dimension of tiles; 
                    output shape will be (patch_size, patch_size, 3)
        tissue_proportion_cutoff: minimum proportion of tissue pixels for export
        img_format
        max_exports: maximum number of images exported per slide
        max_crops: maximum tissue mask crops attempted before 
                   moving on to next slide
    """
    start_time = time.time()
    div = round(self.slide_dims[0]/self.tissue_mask.size[0])
    crops, exports = 0, 0
    while exports < max_exports and crops < max_crops:
      x = random.randint(100, self.slide_dims[0] - patch_size - 100)
      y = random.randint(100, self.slide_dims[1] - patch_size - 100)
      mask_tile = self.tissue_mask.crop((x//div, y//div,
                                         (x+patch_size)//div, 
                                         (y+patch_size)//div))
      mask_tile = np.array(mask_tile).astype('uint8')
      tissue_proportion = (np.count_nonzero(mask_tile)/mask_tile.size)
      crops += 1
      if tissue_proportion > tissue_proportion_cutoff:
        file_name = '{0}-{1}.{2}'.format(self.slide_id, exports, img_format)
        tile = self.slide.read_region((x, y), 0, (patch_size,patch_size))
        tile = Image.fromarray(np.array(tile)[:,:,:3])
        tile_np = np.array(tile)
        if (min_pixel_mean < tile_np.mean() < max_pixel_mean 
            and tile_np.min() < max_pixel_min):
          tile.save(os.path.join(self.export_dir, file_name))
          exports += 1
    print('  time taken: {:0.2f} min'.format((time.time()-start_time)/60))
    print('  exports/crops: {}/{}'.format(exports,crops) )


# ———————————————————————————————————————————————————————————————————

def check_slides_have_resolution(slide_list, resolution):
  print('\nChecking that all slides in slide list have '
        '{}x resolution'.format(resolution))
  removed = 0
  for slide in slide_list:
    try:
      obj = openslide.OpenSlide(slide)
      objective_power = int(obj.properties['openslide.objective-power'])
      level_to_objective = {
        i: round(objective_power/obj.level_downsamples[i], 2)
        for i in range(obj.level_count)}
      objective_to_level = {v: k for k, v in level_to_objective.items()}
      try:
        level = objective_to_level[resolution]
      except KeyError:
        if objective_power < resolution:
          print('removing {}, max resolution {}x'.format(os.path.basename(slide),
                                                     objective_power))
          slide_list.remove(slide)
          removed += 1
    except openslide.lowlevel.OpenSlideUnsupportedFormatError as e:
      print(os.path.basename(slide), e)
  if removed > 0:
    print('removed {} slide from list'.format(removed))
  else:
    print('all slides okay')
  return slide_list


def process_slide(idx, slide_list, process_kwargs):
  slide_path = slide_list[idx]
  try:
    slide = SlideTiling(slide_path, 
                        index='{}/{}'.format(idx, len(slide_list)), 
                        **process_kwargs)
                         
    if FLAGS.mask_save_dir is not None:
      slide.view_mask_overlaid_on_slide(FLAGS.mask_save_dir)
    
    stride = FLAGS.stride
    tissue_proportion = FLAGS.tissue_proportion
    
    if FLAGS.adjust_stride == True:  
      if slide.tissue_area < 1000000:
        stride = FLAGS.stride * 0.75
        tissue_proportion = tissue_proportion - 0.2
      elif slide.tissue_area > 10000000:
        stride = FLAGS.stride * 1.5
      
    export_kwargs = {'patch_size': FLAGS.patch_size,
                   'max_exports': FLAGS.max_exports, 
                   'img_format': FLAGS.img_format,
                   'max_crops': FLAGS.max_exports*10,
                   'stride': stride,
                   'xstride': FLAGS.stride,
                   'ystride': FLAGS.stride,
                   'export_categories': (FLAGS.export_categories.split(',')
                                         if FLAGS.export_categories else None),
                   'dummy_round': FLAGS.dummy_round,
                   'annot_proportion_cutoff': FLAGS.annotation_proportion,
                   'tissue_proportion_cutoff': tissue_proportion,
                   'use_coordinates': FLAGS.use_coordinates,
                   }
    if FLAGS.sampling == 'random':
      exports = slide.export_tiles_random(**export_kwargs)
    elif FLAGS.annotation_dir is not None:
      exports = slide.export_tiles_with_annotations(**export_kwargs) 
    else:
      exports = slide.export_tiles_grid(**export_kwargs)
    return exports

  except (IndexError, ctypes.ArgumentError, FileNotFoundError, ValueError,
          openslide.lowlevel.OpenSlideUnsupportedFormatError,
          openslide.lowlevel.OpenSlideError) as e:
    print('\n*** unable to process {} due to {}'.format(
                                                 os.path.basename(slide_path), 
                                                 e))
    return 0


def process_folder(FLAGS):
  print('saving tiles to:', FLAGS.export_dir)
  misc.verify_dir_exists(FLAGS.export_dir) 
  if FLAGS.mask_save_dir is not None: 
    misc.verify_dir_exists(FLAGS.mask_save_dir)
  
  date_string = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
  #log_file = os.path.join(FLAGS.export_dir, 
  #                        'tiling_logs_{}.txt'.format(date_string))
  #misc.init_output_logging(log_file)
  print('\n', FLAGS)
  
  if FLAGS.slide_list_file is not None:
    if FLAGS.slide_list_file.endswith('.txt'):
      with open(FLAGS.slide_list_file) as f:
        slide_list = [line.rstrip() for line in f]
        # make sure items in slide_list have 'svs' extension
        if slide_list[0].endswith('txt'):
          slide_list = [x.replace('txt', 'svs') for x in slide_list]
    elif FLAGS.slide_list_file.endswith(('pkl', 'pickle')): 
      slide_list = pickle.open(FLAGS.slide_list_file)

    if not os.path.isabs(slide_list[0]):
      print('\nSlide list does not have full paths - will complete using', 
            FLAGS.slide_dir)
      slide_list = [os.path.join(FLAGS.slide_dir, x) for x in slide_list]

    if FLAGS.annotation_dir is not None:
      n_original_slide_list = len(slide_list)
      annotation_list = [os.path.splitext(x)[0] 
                         for x in os.listdir(FLAGS.annotation_dir)]
      slide_list = [x for x in slide_list if os.path.basename(x).split('.')[0]
                    in annotation_list]
      print('{}/{} of slides in slide list with annotations'.format(
            len(slide_list), n_original_slide_list))

  elif FLAGS.slide_dir is not None:
    slide_list = []
    for ext in ['svs', 'tiff']:
      slide_list.extend(glob.glob(os.path.join(FLAGS.slide_dir, '*' + ext)))
    if FLAGS.annotation_dir is not None:
      n_original_slide_list = len(slide_list)
      annotation_list = [os.path.splitext(x)[0] 
                         for x in os.listdir(FLAGS.annotation_dir)]
      slide_list = [x for x in slide_list if os.path.basename(x).split('.')[0]
                    in annotation_list]
      print('{}/{} of slides in slide directory with annotations'.format(
            len(slide_list), n_original_slide_list))

  else:
    print('Unable to load a slide list - quitting program')
    sys.exit(0)
 
  if FLAGS.check_resolution:
    slide_list = check_slides_have_resolution(slide_list, FLAGS.resolution)
  
  print('\nWill process {} slides:'.format(len(slide_list)))
  
  process_kwargs = {'export_dir': FLAGS.export_dir,
                          'mask_dir': FLAGS.mask_dir, 
                          'annotation_dir': FLAGS.annotation_dir,
                          'resolution': FLAGS.resolution, 
                          }

  if FLAGS.n_workers > 1:
    # tiling with pooling
    pool = Pool(FLAGS.n_workers)
    total_exports = pool.map(functools.partial(process_slide, 
                                     slide_list=slide_list, 
                                     process_kwargs=process_kwargs),
                             range(len(slide_list)))
    pool.close(); pool.join()
  
  else:
    # tiling without pooling, mainly for debugging
    total_exports = 0
    #slide_list= [os.path.join(FLAGS.slide_dir, 'MM_sarc_VS11_18879_A1.svs')]
    for idx, slide in enumerate(slide_list):
      exports = process_slide(idx=idx, slide_list=slide_list, 
                              process_kwargs=process_kwargs),
      total_exports += sum(exports)

  print('\nTOTAL EXPORTS: {}'.format(sum(total_exports)))

# ———————————————————————————————————————————————————————————————————

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--slide_dir', required=True, default=None,
      help='needs to either be absolute path or relative to the root_dir')
  parser.add_argument('--export_dir', required=True)
  parser.add_argument('--annotation_dir', default=None)
  parser.add_argument('--mask_dir', default=None)
  parser.add_argument('--mask_save_dir', default=None)
  parser.add_argument('--slide_list_file', default=None)

  parser.add_argument('--sampling', choices=['grid', 'random'], default='grid')
  parser.add_argument('--stride', type=float, default=1.,
      help='step between images for grid tiling')
  parser.add_argument('--adjust_stride', action='store_true', default=False,
      help='increase stride between tiles for slides with very '
           'large tissue area')
  parser.add_argument('--max_exports', type=int, default=999999,
      help='maximum image patches to extract per slide')
  
  parser.add_argument('--resolution', default=40, type=int,
      help='resolution at which slide will be tiled')
  parser.add_argument('--check_resolution', action='store_true')
  parser.add_argument('--patch_size', type=int, default=512,
      help='size of the image sides')
  
  parser.add_argument('--img_format', default='png', 
                      choices=['png', 'jpg', 'jpeg', 'tiff'])
  parser.add_argument('--dummy_round', action='store_true',
      help='runs through tiling without saving images in order to '
           'give number of exports')
  parser.add_argument('--export_categories', default='tumor',
      help='annotation categories to be exported, can have multiple '
           'separated by a comma')
  parser.add_argument('--use_coordinates', dest='use_coordinates',
      type=lambda x:bool(util.strtobool(x)), default=True, 
      help='use coordinates in tile file names, rather then just numbering'
           ' sequentially by exports')
  
  parser.add_argument('-n', '--n_workers', type=int, default=48,
      help='number of workers for multipool')

  parser.add_argument('--annotation_proportion', default=0.75, type=float,
      help='proportion of tile required to have annotation for export')
  parser.add_argument('--tissue_proportion', default=0.75, type=float,
      help='proportion of tile required to have tissue for export')

  FLAGS = parser.parse_args()
  
  cwd = os.getcwd()
  if ROOT_DIR != cwd:
    print('NOTE: root_dir is {}, while working directory is {}'.format(
                                                                ROOT_DIR, cwd))

  for f in ['export_dir', 'annotation_dir', 'slide_dir', 'mask_dir', 
            'slide_list_file', 'mask_save_dir']:
    try:
      arg = vars(FLAGS)[f]
      if not os.path.isabs(arg):
        vars(FLAGS)[f] = os.path.join(ROOT_DIR, arg)
    except TypeError: pass

  start_time = time.time()
  process_folder(FLAGS)
  print('\nTOTAL TIME: {} min'.format((time.time() - start_time)//60))



