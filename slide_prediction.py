import os
import argparse
import glob
from pathlib import Path
import pickle
import random
import statistics

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import misc
import copy

# ——————————————————————————————————————————————————————————————————————

def get_label_from_filename(x):
  if x.startswith(('BM_spin', 'ext_benign')):
    return 'benign'
  elif x.startswith(('MM_sarc', 'boston_meso')):
    return 'tumor'

label_to_value = {'benign': 0.,
                  'tumor': 1.}

def get_value_from_filename(x):
  label = get_label_from_filename(x)
  return label_to_value[label]

def np_to_str(x):
  return np.array_str(x)[1:-1]

def print_run_separator(idx):
  print('\n===============================================================')
  print('RUN {}'.format(idx))

def print_predictions(preds, labels, cases):
  print('\nTest predictions:')
  for i in range(len(preds)):
    print('{}:\t{:.03}\t({})'.format(cases[i], preds[i], int(labels[i])) )
 

def plot_roc_curve_test(preds, labels, folder=None, test_set=None):
  preds = preds.flatten()
  labels = labels.flatten()
  labels.astype('int')
  fpr, tpr, _ = metrics.roc_curve(labels, preds)
  plt.title('Receiver Operating Characteristic')
  plt.plot(fpr, tpr,) #'b', label = 'AUC = %0.2f' % roc_auc)
  #plt.legend(loc = 'lower right')
  #plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([-0.01, 1.])
  plt.ylim([0, 1.01])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  if folder:
    save_path=os.path.join(folder, test_set + '.png')
    plt.savefig(save_path)
    print('saved AUC curve as:', save_path)
  #plt.show()


def plot_roc_curve_crossval(preds, labels, folder=None):
  plt.title('Receiver Operating Characteristic')
  for i,_ in enumerate(preds):
    run_preds = preds[i].flatten()
    run_labels = labels[i].flatten()
    run_labels.astype('int')
    fpr, tpr, _ = metrics.roc_curve(run_labels, run_preds)
    plt.plot(fpr, tpr, label='Cross-val run {}'.format(i+1)) #'b', label = 'AUC = %0.2f' % roc_auc)
  plt.legend(loc = 'lower right')
  #plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([-0.01, 1.])
  plt.ylim([0, 1.01])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  if folder:
    save_path=os.path.join(folder, 'crossval.png')
    plt.savefig(save_path)
    print('saved AUC curve as:', save_path)
  #plt.show()


# ——————————————————————————————————————————————————————————————————————

class EvaluatePredictions():
  def __init__(self, preds, labels, threshold=0.99, print_output=True, 
               n_iter=1000):
    if type(preds) is np.ndarray:
      self.preds = preds.flatten().tolist()
    else: 
      self.preds = preds
    if type(labels) is np.ndarray:
      self.labels = labels.flatten().tolist()
    else:
      self.labels = labels
    self.threshold = threshold
    self.print_output = print_output
    self.n_iter = n_iter    
    self.size = len(preds)

  def sample(self, list_):
    # Following three rules:
    #    1. Sampling with replacement
    #    2. Equal probability for all samples
    #    3. Same size as original list
    if type(list_) is np.ndarray:
      sample_ = np.random.choice(list_, size=self.size, replace=True)
    else:
      sample_ = random.choices(list_, k=self.size)
    return sample_
  
  def binarize_predictions(self, preds):
    x = copy.deepcopy(preds)
    x[x > self.threshold] = 1.
    x[x <= self.threshold] = 0.
    return x

  def specificity(self, labels, preds):
    cm = metrics.confusion_matrix(labels, preds)
    return cm[0,0]/(cm[0,0]+cm[0,1])

  def run(self):
    dict_ = {x: [] for x in ['auc', 'acc', 'sensitivity', 'specificity']}
    data = list(zip(self.preds, self.labels))
    
    for idx in range(self.n_iter):
      if idx == 0:
        preds, labels = zip(*data)
      else:
        sample_ = self.sample(data)
        preds, labels = zip(*sample_)
      dict_['auc'].append(metrics.roc_auc_score(labels, preds))
      binary = self.binarize_predictions(np.array(preds))
      dict_['acc'].append(metrics.accuracy_score(labels, binary))
      dict_['sensitivity'].append(metrics.recall_score(labels, binary)) # sens
      dict_['specificity'].append(self.specificity(labels, binary)) # sens
      #dict_['precision'].append(metrics.precision_score(labels, binary)) # PPV
      #dict_['f1'].append(metrics.f1_score(labels, binary))

    if self.print_output:
      print('\n\nMETRICS:')
      for key, value in dict_.items():
        print('{}: {:.03} ({:.03}-{:.03})'.format(key, value[0], 
              np.percentile(value, 5), np.percentile(value, 95)))


def threshold_search(preds, labels):
  threshold_values = [0.5, 0.80, 0.90, 0.95, 0.98, 0.99, 0.99, 0.999, 0.9999,
                      0.99999]
  best_acc = 0
  best_threshold = 0
  for threshold in threshold_values:
    binary = binarize_predictions(preds, threshold)
    acc = metrics.accuracy_score(labels, binary)
    precision = metrics.precision_score(labels, binary) # PPV
    recall = metrics.recall_score(labels, binary) # sensitivity
    f1 = metrics.f1_score(labels, binary)
    print('\n{} - acc: {:.03}, precision {:.03}, recall {:.03}, '
          'f1 {:.03}'.format(threshold, acc, precision, recall, f1), 
          flush=True)
    if acc > best_acc:
      best_acc = acc
      best_threshold = threshold

  print('\nBEST THRESHOLD: {}'.format(best_threshold))
  return threshold


def save_histogram(data, title, save_dir):
  n, bins, patches = plt.hist(data, bins=100)
  plt.title(title)
  plt.ylabel('N tiles')
  plt.xlabel('value')
  #plt.show()
  plt.savefig(os.path.join(save_dir, title + '.jpg'))
  plt.clf()



# ——————————————————————————————————————————————————————————————————————

class SlidePrediction():
  def __init__(self, subdir, preds_dir, case_split, save_histograms=True,
               test_set='val'):
    self.subdir = subdir
    self.predictions_dir = preds_dir
    self.save_histograms = save_histograms
    self.test_set = test_set
    if save_histograms:
      dirs = os.path.split(subdir)
      self.histogram_dir = os.path.join(dirs[0], 'histograms', dirs[1])
      Path(self.histogram_dir).mkdir(parents=True, exist_ok=True)
    self.case_split = case_split
    self.fv_len = 512 
    self.verbose = False
    self.modes = list(case_split.keys())
    # make dictionary mapping case to mode
    self.case_to_mode = dict( (v,k) for k in self.case_split 
                              for v in self.case_split[k] )
    self.load_tile_predictions()

  def load_tile_predictions(self):
    self.output_dict = {x: {} for x in self.modes}
    pkl_list = [x for x in os.listdir(self.predictions_dir)
                if x.endswith('pkl')]
    for pkl_file in pkl_list: 
      try:
        pred =  misc.load_pkl(os.path.join(self.predictions_dir, pkl_file))
        case_id = os.path.splitext(pkl_file)[0]
        try:
          mode = self.case_to_mode[case_id]
          self.output_dict[mode][case_id] = pred
        except KeyError:
          print('case {} not found in case split'.format(pkl_file))
      except pickle.UnpicklingError as e:
        print('unable to load {} due to:'.format(pkl_file), e)
    n_preds = {mode: len(self.output_dict[mode]) for mode in self.modes}
    print('loaded predictions:', n_preds )
  
  def avg_pool_prediction(self, topN=0.005, min_imgs=10, output='softmax'):
    """ Average pooling prediction using the tiles that are most likely
    to be tumor.

    # Args:
      topN: proportion of tiles to use (ie sort in descending order and 
            then use topN * num_tiles)
      output: type of output data from network to use (ie softmax or fc)
    """
    predictions_dict = {}
    labels_dict = {}
    cases_dict = {}
    for mode in self.modes:
      preds = np.empty(0)      
      labels = np.empty(0)      
      cases = []
      if self.save_histograms:
        set_ = self.test_set if mode=='test' else 'train'
        histogram_subdir = os.path.join(self.histogram_dir, set_)
        Path(histogram_subdir).mkdir(parents=True, exist_ok=True)     
      case_list = list( self.output_dict[mode].keys() )
      for case in case_list:
        try:
          output_data = self.output_dict[mode][case][output][:,1]
          if self.save_histograms:
            save_histogram(output_data, case, histogram_subdir)
          output_data = np.sort(output_data)[::-1]
          if isinstance(topN, int):
            n_imgs = topN
          else:
            n_imgs = int(np.max((len(output_data) * topN, min_imgs)))
          if self.verbose:
            print(case, n_imgs)
          pred = np.mean(output_data[:n_imgs]).reshape(1)
          preds = np.concatenate((preds, pred))
          label = np.array(get_value_from_filename(case)).reshape(1)
          labels = np.concatenate((labels, label)) 
          cases.append(case)
        except KeyError as e:
          print(case, e)
      predictions_dict[mode] = preds
      labels_dict[mode] = labels
      cases_dict[mode] = cases
    return predictions_dict, labels_dict, cases_dict


# ——————————————————————————————————————————————————————————————————————

def evaluate_run(subdir, case_split, prediction_method, threshold,
                 fv_type='pred_features', print_predictions=False):
  """ Evaluate the performance from a single cross validation run
  """  
  preds_dir = os.path.join(subdir, 'predictions_cv/case') 
  predictor = SlidePrediction(subdir, preds_dir, case_split)
  
  modes = list(case_split.keys())

  test_metrics = {}
  preds, labels, cases = predictor.avg_pool_prediction() 
  
  for mode in modes:
    if mode in ['val', 'test']:
      if print_predictions:
        print_predictions(preds[mode], labels[mode], cases[mode])
      if FLAGS.threshold_search and mode=='val':
        threshold = threshold_search(preds[mode], labels[mode])
      metrics = EvaluatePredictions(preds[mode], labels[mode], threshold).run()
      test_metrics = metrics
  return test_metrics, preds, labels


def evaluate_cross_val_runs(run_dir, case_split, threshold, prediction_method,
                            print_predictions=False, **kwargs):
  run_subdirs = glob.glob(os.path.join(run_dir, 'run_*'))
  full_metrics = {}
  cv_preds, cv_labels = [], []
  # loop over all CV runs and save metrics to dict
  for idx in range(1, len(run_subdirs)+1):
    print_run_separator(idx)
    subdir = run_subdirs[idx-1]
    split = {'train': case_split[str(idx)]['train'],
             'val': case_split[str(idx)]['test']}
    full_metrics[idx], preds, labels = evaluate_run(subdir, split, 
                                       prediction_method, threshold, 
                                       print_predictions=print_predictions)
    cv_preds.append(preds['val'])
    cv_labels.append(labels['val'])

  # for each prediction method calculate average metrics, including
  # std dev for AUC
#  print('\nAverage test metrics values:')
#  for idx in full_metrics.keys():
#    run_metrics = np.array(full_metrics[idx]).reshape((-1,1))
#    try:
#      pred_metrics = np.concatenate((pred_metrics, run_metrics), 1)
#      aucs = np.concatenate((aucs, run_metrics[0]))
#    except:
#      pred_metrics = run_metrics
#      aucs= run_metrics[0]
#  avg_metrics = [round(x, 3) for x in np.mean(pred_metrics[1:], 1)]
#  auc_mean = round(statistics.mean(aucs), 3)
#  auc_std = round(statistics.stdev(aucs), 3)
#
  EvaluatePredictions(np.concatenate(cv_preds), np.concatenate(cv_labels), 
                      threshold).run()
  plot_roc_curve_crossval(cv_preds, cv_labels)
  

def test_set_prediction(run_id, test_list, test_pred_dir, prediction_method, 
                        threshold=0.99, test_set='val', print_predictions=False,
                        **kwargs):
  """ Evaluate test set by averaging predictions from models trained 
  in multi-fold cross validation runs
  """
  case_split = {'test': misc.load_txt_file_lines(test_list)}
  run_dir = os.path.join(FLAGS.runs_main, run_id)
  run_subdirs = glob.glob(os.path.join(run_dir, 'run_*'))
  for idx in range(1, len(run_subdirs)+1):
    print_run_separator(idx)
    subdir = os.path.join(run_dir, 'run_{}'.format(idx))    
    
    preds_dir = os.path.join(subdir, test_pred_dir, 'case') 
    predictor = SlidePrediction(subdir, preds_dir, case_split, 
                                test_set=test_set)

    preds, labels, cases = predictor.avg_pool_prediction()
    preds = preds['test'].reshape((-1, 1))
    labels = labels['test'].reshape((-1, 1))
    try:
      full_preds = np.concatenate((full_preds, preds), 1)
      full_labels = np.concatenate((full_labels, labels), 1)
    except NameError:
      full_preds = preds
      full_labels = labels
    
  avg_preds = np.mean(full_preds, 1)
  if print_predictions:
    print_predictions(avg_preds, labels, cases['test'])
  EvaluatePredictions(avg_preds, labels, threshold, n_iter=1000).run()
  plot_roc_curve_test(avg_preds, labels) 


# ——————————————————————————————————————————————————————————————————————

def main(FLAGS):  
  run_dir = os.path.join(FLAGS.runs_main, FLAGS.run_id)
  # load slide split  
  case_split = misc.load_pkl(os.path.join(run_dir, 'case_split.pkl'))

  # path to csv summary file 
  handle = (FLAGS.run_id[:-2] if FLAGS.run_id.endswith('_', -3, -1) 
            else FLAGS.run_id)
  csv_file = os.path.join(FLAGS.runs_main, 
                              'run_summaries/{}.csv'.format(handle))

  from folder_config import folder_config
  folders = folder_config.get(FLAGS.test_set, folder_config['val'])
 
  config = {'run_id': FLAGS.run_id,
            'run_dir': run_dir,
            'case_split': case_split,
            'csv_file': csv_file,
            'cv_idx': FLAGS.cv_idx,
            'threshold': FLAGS.threshold,
            'prediction_method': FLAGS.prediction_method,
            'reset_csv': FLAGS.reset_csv,
            'plot_histograms': FLAGS.plot_histograms,
            'test_list': folders['list'],
            'test_pred_dir': folders['output_dir'],
            'test_set': FLAGS.test_set,
            'print_predictions': FLAGS.print_predictions,
            }
            

  if FLAGS.test_set in ['external', 'referrals']:
    test_set_prediction(**config)
  else:
    evaluate_cross_val_runs(**config)

# ——————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # locate run folder
  parser.add_argument('--runs_main', 
      default='/home/alevine/mesothelioma/results')
  parser.add_argument('--run_id', default=None,
      help="Saved model to use for generating predictions")
  parser.add_argument('--cv_idx', default=None, type=int)

  # options for predictions
  parser.add_argument('--new_fv', action='store_true')
  parser.add_argument('--prediction_method', default='avg_pool')
  parser.add_argument('--threshold_search', action='store_true',
    help='search through values to determine best threshold for considering'
         'a case malignant')
  parser.add_argument('--threshold', default=0.999, type=float,
    help='threshold to consider a case malignant (i.e. for getting accuracy')

  parser.add_argument('--reset_csv', action='store_true',
      help='clear any existing data in csv summary file')  
  parser.add_argument('--plot_histograms', action='store_true')
  parser.add_argument('--print_predictions', action='store_true')
  
  # testing
  parser.add_argument('--test_set', default=None,
      choices=['external', 'referrals', 'val'])

  FLAGS = parser.parse_args()
  
  main(FLAGS)

