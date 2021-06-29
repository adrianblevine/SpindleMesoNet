import os, glob, pickle
import sys

run_ids = sys.argv[1:]

results_main = '/projects/pathology_char/pathology_char_results/mesothelioma/results'

for run in run_ids:
  for subdir in ['run_1', 'run_2', 'run_3']:
    for mode in ['train', 'test']:
      data_dir = os.path.join(results_main, run, subdir, 'rnn_data', mode)
      print(data_dir)
      item_list = glob.glob(os.path.join(data_dir, '*png'))
      for item in item_list:
          with open(item, 'rb') as f:
              data = pickle.load(f)
              data_shape = data['inputs'].shape
              if data_shape[0] != 10:
                  print(os.path.basename(item), data_shape)
                  os.remove(item)
