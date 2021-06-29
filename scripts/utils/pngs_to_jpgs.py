import os
import glob
import functools
from PIL import Image
from multiprocessing import Pool

input_main = '/projects/pathology_char/pathology_char_results/mesothelioma/images_5120'
output_main = '/projects/pathology_char/images_5120_jpg'

def convert_to_jpg(input_path, output_dir):
  img = Image.open(input_path)
  filename = os.path.splitext(os.path.basename(input_path))[0]
  output_path = os.path.join(output_dir, filename + '.jpg')
  img.save(output_path)

subdirs = [x for x in glob.glob(os.path.join(input_main, '*'))
             if os.path.isdir(x)]
print(subdirs, '\n')
for subdir in subdirs:
  print(subdir)
  output_dir = os.path.join(output_main, os.path.basename(subdir))
  if not os.path.isdir(output_dir): os.makedirs(output_dir)
  img_list = glob.glob(os.path.join(subdir, '*png'))
  #for img in img_list:
  #  convert_to_jpg(img, output_dir)
  pool = Pool(48)
  pool.map(functools.partial(convert_to_jpg, output_dir=output_dir),
           img_list)
  pool.close(); pool.join()
