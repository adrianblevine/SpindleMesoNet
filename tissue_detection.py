
import os
import random
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool
import functools

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import skimage as ski
import skimage.color as color
import skimage.morphology as ski_morphology

import cv2
import openslide

import misc

# ———————————————————————————————————————————————————————————————————

working_dir = '/path/to/dir/'

FLAGS=[]
  
class TissueDetection():
  """ Load a slide and generate mask of tissue region

  # Arguments
     slide_path 
  
  # Attributes
      path
      name
      slide
      tissue_mask
      bounding_box

  # Methods
      read_level
      generate_mask
      show_overlay
  """    
  def __init__(self, slide_path):
    self.slide_path = slide_path
    self.name = os.path.basename(slide_path)
    self.slide_obj = openslide.OpenSlide(slide_path)
    self.micron_per_pixel = float(self.slide_obj.properties['openslide.mpp-x'])

  def read_at_downsample(self, downsample):
    """ Read a given level of an openslide object 
    
    # Arguments
      downsample: downsample of slide to read, relative to max resolution
   
    # Returns: RGB image of whole slide as a numpy array  
    """
    # read images downsampled by a factor of 16
    resize = False
    downsamples = self.slide_obj.level_downsamples
    level = np.searchsorted(np.array(downsamples), downsample, side="left")
    # if desired downsample is not present, read from closest largest 
    # level and resize
    if round(downsamples[level]) < downsample: 
      level -= 1
      resize = True
    mpp = round(downsample * self.micron_per_pixel)

    im = self.slide_obj.read_region((0,0), level, 
                        self.slide_obj.level_dimensions[level]).convert('RGB')
    if resize:
      target_size = tuple(round(x/downsample) 
                          for x in self.slide_obj.dimensions)
      im = im.resize(target_size)
    im = np.array(im)
    return im, mpp


  def generate_mask(self, downsample=16, make_contour=False,
                       return_mask=False, apply_laplacian=True, **kwargs):
    """ Method described by Strom et al, Lancet Oncol, 2020

    DOI:https://doi.org/10.1016/S1470-2045(19)30738-7

    """
    
    warnings.filterwarnings('ignore')
    im, mpp = self.read_at_downsample(downsample)
    # converted the images from RGB to grayscale
    grey_img = color.rgb2grey(im)


     
    # A filter approximating the 2D Laplacian operator 
    if apply_laplacian:
      lap = ski.filters.laplace(grey_img, ksize=3)
    else:
      lap = grey_img

    # threshold absolute magnitude of the resulting response 
    # using Otsu’s method 
    threshold = ski.filters.threshold_otsu(lap)
    mask = np.zeros(lap.shape)
    mask[lap < threshold] = 1
    
     # set the edges of the slide to 0
    xdim, ydim = mask.shape
    xclip, yclip = int(xdim*0.01), int(ydim*0.01)
    mask[:xclip,:] = 0
    mask[-xclip:,:] = 0
    mask[:,:yclip] = 0
    mask[:,-yclip:] = 0

    # morphological closing with a disk-shaped structuring element 
    # having a radius of 50 μm 
    element_size = round(2*50/mpp)
    str_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (element_size, element_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, iterations=1,
                            kernel=str_element)
    
    
    # catch slides where Laplacian filter doesn't adequately separate 
    # tissue and instead apply otsu on greyscale image
    fraction = round(100*np.sum(mask)/mask.size, 5)
    
    if fraction > 70 or fraction < 8:
      print(os.path.basename(self.slide_path), fraction, flush=True)
      threshold = ski.filters.threshold_otsu(grey_img)
      mask = np.zeros(grey_img.shape)
      mask[grey_img < threshold] = 1
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, iterations=1, 
                              kernel=str_element)

    # filling of holes and removal of  objects  having  an  area  
    # smaller  than  100  000  μm2. 
    min_size = round(100000/mpp**2)
    labels, num_labels = ski.measure.label(mask, connectivity=2, 
                                             return_num=True)
    labels = ski_morphology.remove_small_objects(labels, min_size)
    mask = ski_morphology.remove_small_holes(labels, min_size)
    mask[mask > 0] = 1

    # HSV transform and excluding any objects whose mean hue was less than 0.7. 
    hue = color.rgb2hsv(im)[:,:,0]
    labels, num_labels = ski.measure.label(mask, connectivity=2, 
                                           return_num=True)
    for label in range(1, num_labels + 1):
      mean_hue = np.mean(hue[labels == label] + 1e-8)
      if mean_hue < 0.68:
        mask[labels == label] = 0

    self.tissue_area = np.sum(mask)
    self.tissue_mask = Image.fromarray(mask.astype('uint8'), mode='L')
    self.bounding_box = tuple([int(x*downsample) 
                               for x in self.get_bounding_box(mask)])

    
    if make_contour:
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
      contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel=kernel)
      self.contour = Image.fromarray(contour.astype('uint8'), mode='L')

    warnings.filterwarnings('default')
    
    if return_mask:
      return self.tissue_mask
 

  def get_bounding_box(self, mask):
    """ Determines the corners of a rectangle that completely contains 
        a given mask  
    
    # Note:
        this function returns coordinates in the axes that they are used by 
        PIL/openslide, however x and y need to be flipped in order to be      
        used in numpy/matplotlib
    
    # Args:
        mask: a numpy array

    # Returns: a tuple
     """ 
    xax = np.amax(mask, axis = 1)
    yax = np.amax(mask, axis = 0)
    xmin = np.argmax(xax)
    xmax = mask.shape[0] - np.argmax(np.flip(xax, 0))
    ymin = np.argmax(yax)
    ymax = mask.shape[1] - np.argmax(np.flip(yax, 0))
    return (ymin, ymax, xmin, xmax)


  def show_overlay(self, save_path=None, size=10, overlay='mask'):
    """ Generates an overlay of the slide and tissue region (either
    as mask or contours) and then either displays or saves it 
    """
    roi = self.slide_obj.get_thumbnail((5000, 5000))
    if overlay == 'mask':
      mask = self.tissue_mask.resize(roi.size)
    elif overlay == 'contour':
      mask = self.contour.resize(roi.size)
    if roi.size[1] > roi.size[0]:
      roi = roi.rotate(90, expand=True)
      mask = mask.rotate(90, expand=True)

    ratio = int(roi.size[0]/roi.size[1])
    roi = np.asarray(roi)
    mask = np.asarray(mask).astype('float32')
    
    fig = plt.figure(figsize = (size, size*ratio))
    plt.suptitle((self.name, self.slide_obj.dimensions, self.bounding_box))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    plt.imshow(roi, interpolation='none')
    if overlay == 'mask':
      plt.imshow(mask, 'viridis', interpolation='none', alpha=0.3)
    elif overlay == 'contour':
      plt.imshow(mask, 'binary', interpolation='none', alpha=0.5)
    fig.tight_layout()
    if save_path:
      plt.savefig(save_path, orientation='landscape')
      plt.close()
    else:
      plt.show()


def process_slide(slide, FLAGS):
    if not os.path.isabs(slide):
      slide = os.path.join(FLAGS.slide_dir, slide)
    try:
      td = TissueDetection(slide)
      td.generate_mask(apply_laplacian=True, make_countour=False)
      save_path = os.path.join(FLAGS.save_dir, 
                               os.path.basename(slide)[:-4])
      if FLAGS.save is not None:
        if FLAGS.save == 'mask':
          td.tissue_mask.save(save_path + '.tif')
        elif FLAGS.save == 'image':
          td.show_overlay(save_path + '.jpg', overlay=FLAGS.overlay)
      else:
        td.show_overlay()
      del td
    except IndexError as e:
      print('*** unable to process %s *** due to:' % os.path.basename(slide), e )


def process_slide_list(FLAGS): 
  """ Running the program as a script will iterate over every slide in a folder,
  showing first the downsampled slide with overlaid tissue region.
  """

  Path(FLAGS.save_dir).mkdir(parents=True, exist_ok=True)
  
  if FLAGS.slide_list:
    slide_list = misc.load_txt_file_lines(FLAGS.slide_list)
    slide_list = [x.replace('txt', 'svs') for x in slide_list] 
  else:
    slide_list = os.listdir(FLAGS.slide_dir)
  random.shuffle(slide_list)
  
  if FLAGS.number:
    slides = slide_list[:FLAGS.number]
  else: 
    slides = slide_list
 
  print('\nwill process {} slides'.format(len(slides))) 
  if FLAGS.n_workers == 1:
    for slide in slides:
      process_slide(slide, FLAGS)
  else: 
    pool = Pool(FLAGS.n_workers)
    pool.map(functools.partial(process_slide, FLAGS=FLAGS), 
             slide_list)
    pool.close(); pool.join()

# ———————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--number', type=int, default=None,
        help='Number of slides to view')
  parser.add_argument('--save_images', action='store_true',
        help='Save images to file (vs. just displaying')
  parser.add_argument('--overlay', choices=['mask','contour'], default='mask',
        help='Display tissue area as a contoured outline or mask')
  parser.add_argument('--save', choices=['mask', 'image'], default=None)
  parser.add_argument('--slide_dir') 
  parser.add_argument('--save_dir', 
      default=os.path.join(working_dir, 'tissue_mask_images'))
  parser.add_argument('-n', '--n_workers', type=int, default=1)
  parser.add_argument('--slide_list', default=None)
  
  FLAGS, unparsed = parser.parse_known_args()
  
  for f in ['slide_dir', 'save_dir' ]:
    try:
      arg = vars(FLAGS)[f]
      if not os.path.isabs(arg):
        vars(FLAGS)[f] = os.path.join(working_dir, arg)
    except TypeError: pass
  
  # set Agg backend to avoid tkinter error when running in srun 
  if FLAGS.n_workers > 1:
    plt.switch_backend('Agg')

  print(FLAGS)
  process_slide_list(FLAGS)


