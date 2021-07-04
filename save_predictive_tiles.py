import os
import sys
import shutil
import argparse
import glob

import numpy as np

import misc
from slide_prediction import get_label_from_filename

results_main = '/path/to/dir/results'

parser = argparse.ArgumentParser()
parser.add_argument('--run_id')
parser.add_argument('--cases_file', default='/path/to/dir/lists/cases.txt')
parser.add_argument('--most_benign', action='store_true')
parser.add_argument('--ru', default=None)
args = parser.parse_args()

cases = misc.load_txt_file_lines(args.cases_file)
run_dirs = glob.glob(os.path.join(results_main, args.run_id, 'run_*'))

if args.most_benign:
  output_main = os.path.join(results_main, args.run_id, 'top_benign_imgs')
else:
  output_main = os.path.join(results_main, args.run_id, 'top_malignant_imgs')

misc.verify_dir_exists(output_main)

# define image_folders
img_dirs = {
  'benign': '/path/to/dir/images/benign',
  'tumor': '/path/to/dir/images/tumor'}

for case in cases:
  print(case)
  # load tile predictions
  label = get_label_from_filename(case)
  dict_ = {}
  for dir_ in run_dirs:
    file_ = os.path.join(dir_, 'predictions_test/image', case + '.pkl')
    data = misc.load_pkl(file_)
    for img in data.keys():
      softmax = data[img]['softmax']
      try:
        dict_[img].append(softmax[1])
      except KeyError:
        dict_[img] = [softmax[1]]
  # average predictions across 5 runs
  list_ = [(img, np.mean(dict_[img])) for img in dict_.keys()]

  # rank tiles by prediction value
  list_.sort(key=lambda x: x[1])
  if not args.most_benign:
    list_.reverse()

  # copy image to folder
  output_dir = (os.path.join(output_main, case))
  misc.verify_dir_exists(output_dir)
  for img in list_[:10]:
    path = os.path.join(img_dirs[label], img[0] + '.png')
    shutil.copy(path, output_dir) 




