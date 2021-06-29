import os
import pickle
import sys
import shutil

results_main = '/projects/pathology_char/pathology_char_results/mesothelioma/results'

def fix_case_split(run_id):
  pt_split_path = os.path.join(results_main, run_id, 'case_split.pkl')
  copy_path = os.path.join(results_main, run_id, 'COPY_case_split.pkl')
  shutil.copy(pt_split_path, copy_path)

  with open(pt_split_path, 'rb') as f:
    split = pickle.load(f)

  for run in split.keys():
    for mode in split[run].keys():
      split[run][mode] = ['MM_sarc_VR14_88' if x == 'MM_sarc_VR1488_A4' else
                          x for x in split[run][mode]]

  with open(pt_split_path, 'wb') as f:
    pickle.dump(split, f, protocol=pickle.DEFAULT_PROTOCOL)


def fix_image_split(run_id):
  img_split_path = os.path.join(results_main, run_id, 'img_split.pkl')
  copy_path = os.path.join(results_main, run_id, 'COPY_case_split.pkl')
  shutil.copy(img_split_path, copy_path)

  with open(img_split_path, 'rb') as f:
    split = pickle.load(f)
  for run in split.keys():
    for mode in split[run].keys():
      for item in split[run][mode]:
        if 'VR1488' in item:
          print(item)

  #with open(img_split_path, 'wb') as f:
  #  pickle.dump(split, f, protocol=pickle.DEFAULT_PROTOCOL)




if __name__ == "__main__":
  if len(sys.argv) > 1:
    run_ids = sys.argv[1:]
  else:
    run_ids = []
  
  for run_id in run_ids:  
    print(run_id)
    fix_case_split(run_id)
    fix_image_split(run_id)
