  def make_feature_vectors(self, topN=0.005):
    self.fvs_dict = {}
    self.labels_dict = {}
    for mode in ['train', 'test']:
      labels = np.empty(0)      
      
      case_list = list(self.output_dict[mode].keys())
      for case in case_list:
        try:
          output_data = self.output_dict[mode][case]['softmax'][:,1]
          output_data = np.sort(output_data)[::-1]
          # make dictionary for feature vector values
          dict_ = {}
          len_ = len(output_data)
          # top value
          dict_['max'] = output_data[0]
          # top avg pool
          n_imgs = int(np.max((len(output_data) * topN, 3)))
          dict_['top_avg'] = np.mean(output_data[:n_imgs])
          # avg value
          dict_['avg'] = np.mean(output_data)
          # N above 0.5
          dict_['>0.5'] = np.sum(output_data > 0.5)/len_
          dict_['>0.6'] = np.sum(output_data > 0.6)/len_
          dict_['>0.7'] = np.sum(output_data > 0.7)/len_
          # N above 0.9
          dict_['>0.8'] = np.sum(output_data > 0.8)/len_
          dict_['>0.9'] = np.sum(output_data > 0.9)/len_
          dict_['>0.95'] = np.sum(output_data > 0.95)/len_
          dict_['>0.99'] = np.sum(output_data > 0.99)/len_
          dict_['len'] = len_
          # convert to numpy array and concatenate
          fv = np.fromiter(dict_.values(), dtype=float).reshape(1, -1)
          try: 
            fvs = np.concatenate((fvs, fv))
          except (NameError, ValueError):
            fvs = fv

          label = np.array(get_value_from_filename(case)).reshape(1)
          labels = np.concatenate((labels, label)) 
        except KeyError as e:
          print(case, e)
      
      self.fvs_dict[mode] = fvs
      self.labels_dict[mode] = labels
      # need to delete fvs variable so it doesn't continue from train to test
      del fvs
    #return self.fvs_dict, self.labels_dict


  def load_cnn_features(self, topN=50, dimensionality_reduction=True):
    if not FLAGS.new_fv:
      try:
        self.fvs_dict = misc.load_pkl(os.path.join(self.subdir, 
                                                 'fvs_dict.pkl'))    
        self.labels_dict = misc.load_pkl(os.path.join(self.subdir, 
                                                    'labels_dict.pkl'))    
      except:
        self.get_top_cnn_features(topN, dimensionality_reduction)
    else:
      self.get_top_cnn_features(topN, dimensionality_reduction)

    print('feature vectors loaded') 


  def get_top_cnn_features(self, topN=50,  
                              dimensionality_reduction=True):
    self.fvs_dict = {}
    self.labels_dict = {}
    for mode in ['train', 'test']:
      #fvs = np.empty((0, topN*self.fv_len))      
      labels = np.empty(0)      
      
      case_list = list(self.output_dict[mode].keys())
      for case in case_list:
        print(case)
        try:
          label = np.array(get_value_from_filename(case)).reshape(1)
          labels = np.concatenate((labels, label))
          softmax = self.output_dict[mode][case]['softmax'][:,1]
          softmax = softmax.reshape((-1,1))
          fv = self.output_dict[mode][case]['fv']
          
          # zip softmax and feature vectors together 
          dict_ = {np.array_str(softmax[i])[1:-1]: fv[i] 
                  for i in range(len(softmax))}
          z = np.array(list(zip(softmax,fv)))
          # rank feature vectors by softmax value and unzip
          z = z[z[:,0].argsort()]
          softmax, full_fv = zip(*(tuple(z)))
   
          # verify that softmax and fv are still properly matched up
          for i in random.sample(range(0, len(z)),100):
            k = np_to_str(z[i][0])
            v = z[i][1]
            #if not (dict.get(k) == v).all(): print(k)
            #if dict.get(k) is not None:
            #  assert (dict.get(k) == v).all(), print( (dict.get(k) == v))
          
          # take top 100 feature vectors
          full_fv = np.array(full_fv)
          fv = full_fv[-topN:, :]
          if dimensionality_reduction:
            pca = decomposition.PCA(n_components=10)
            fv = pca.fit_transform(fv)
          # reshape and concatenate full fv array
          fv = fv.ravel()
          fv = np.expand_dims(fv, 0)
          try:
            self.fvs_dict[mode] = np.concatenate((self.fvs_dict[mode], fv))
          except (NameError, KeyError):
            self.fvs_dict[mode] = fv
        except (IndexError, KeyError) as e:
          print('*** {} ***'.format(e))
          print(self.output_dict[mode][case])
          self.output_dict[mode].pop(case)
      #self.fvs_dict[mode] = fvs
      self.labels_dict[mode] = labels.reshape((-1,1))
    misc.save_pkl(self.fvs_dict, os.path.join(self.subdir, 'fvs_dict.pkl'))  
    misc.save_pkl(self.labels_dict, os.path.join(self.subdir, 
                                                 'labels_dict.pkl'))    
 
  def KNN_prediction(self):
    self.predictions_dict = {}
    clf = neighbors.KNeighborsClassifier()
    clf.fit(self.fvs_dict['train'], self.labels_dict['train'])
    preds = clf.predict_proba(self.fvs_dict['test'])
    self.predictions_dict['test'] = preds[:,1] 
    return self.predictions_dict, self.labels_dict


  def SVM_prediction(self):
    self.predictions_dict = {}
    clf = svm.SVC(probability=True)
    # use train feature vectors to train classifier
    clf.fit(self.fvs_dict['train'], self.labels_dict['train'])
    # evaluate on test feature vectors and output predictions
    preds = clf.predict_proba(self.fvs_dict['test'])
    self.predictions_dict['test'] = preds[:,1] 
    return self.predictions_dict, self.labels_dict

  def random_forest_prediction(self):
    self.predictions_dict = {}
    clf = ensemble.ExtraTreesClassifier()
    X, y = utils.shuffle(self.fvs_dict['train'], self.labels_dict['train'])
    clf.fit(X, y)
    #print(X, y)
    preds = clf.predict_proba(self.fvs_dict['test'])
    print(preds, self.labels_dict['test'])
    self.predictions_dict['test'] = preds[:,1]
    return self.predictions_dict, self.labels_dict

  def max_pool_prediction(self):
    self.predictions_dict = {}
    self.labels_dict = {}
    for mode in ['train', 'test']:
      preds = np.empty(0)      
      labels = np.empty(0)      

      for case in self.output_dict[mode].keys():
        try:
          softmax = self.output_dict[mode][case]['softmax'][:,1]
          pred = np.sort(softmax)[-1].reshape(1)
          preds = np.concatenate((preds, pred))
          label = np.array(get_value_from_filename(case)).reshape(1)
          labels = np.concatenate((labels, label))
        except KeyError as e:
          print(case, e)
      self.predictions_dict[mode] = preds
      self.labels_dict[mode] = labels
    return self.predictions_dict, self.labels_dict


  def plot_histograms(self, histogram_dir, output='softmax'):
    # make lists of train and test values
    for mode in ['test', 'train']:
      print('\n{}'.format(mode.upper()))
      values_dict = {'softmax': {'benign': np.empty((0,2)),
                                 'tumor': np.empty((0,2))},
                     'fc': {'benign': np.empty((0,2)),
                            'tumor': np.empty((0,2))}
                     }

      for case in self.output_dict[mode].keys():
        label = get_label_from_filename(case) 
        softmax = self.output_dict[mode][case]['softmax'][:,1]

        for output in ['softmax', 'fc']:
          values = self.output_dict[mode][case][output]
          values_dict[output][label]= np.concatenate(
                              (values_dict[output][label],
                               values))
       
      for category in ['benign', 'tumor']:
        values = values_dict[output][category][:,1]
        title = '{}-{}'.format(mode, category)
        save_histogram(values, title, histogram_dir)





# ——————————————————————————————————————————————————————————————————————
def evaluate_single_run(cv_idx, run_id, run_dir, case_split, prediction_method, 
                        threshold, plot_histograms, reset_csv, csv_file,
                        **kwargs):
  # predict and evaluate on just a single run
  subdir = os.path.join(run_dir, 'run_' + str(cv_idx))
  try: 
    case_split = case_split[str(cv_idx)]
  except:
    case_split = case_split
  
  if plot_histograms:
    plot_histograms(subdir, case_split)

  else:
    metrics = evaluate_run(subdir, case_split, prediction_method, 
                           threshold)
    
    # write to summary csv file
    if FLAGS.reset_csv:
      if os.path.exists(csv_file): os.remove(csv_file)
    
    try: 
      with open(csv_file) as f:
        n_lines = (len(f.readlines()))
    except: n_lines = 0

    with open(csv_file, 'a', newline='') as f:
      w = csv.writer(f, delimiter='\t')
      # write headers if empty
      if n_lines == 0:
        header = ['run_id', 'prediction_method', 'auc', 'acc', 'precision', 
                  'recall', 'f1']
        w.writerow(header)
      # write output to file
      for k in metrics.keys():
        line = [run_id, k] + [round(x, 5) for x in (metrics[k])]
        w.writerow(line)


def multiple_model_prediction(models, prediction_method,
                         fv_type='pred_features', threshold=0.99, **kwargs):
  """ Bootstrap by averaging predictions from multiple specified models 
  """
  for i, model in enumerate(models):
    print('\n===============================================================')
    print(model)
    run_dir = os.path.join(FLAGS.runs_main, model)
    case_split = misc.load_pkl(os.path.join(run_dir, 'case_split.pkl'))
    subdir = os.path.join(run_dir, 'run_1')    
    preds_dir = os.path.join(subdir, 'predictions_case') 
    print(case_split['test']) 
    predictor = SlidePrediction(subdir, preds_dir, case_split)
  
    if prediction_method in ['svm','knn', 'random_forest']:
      if fv_type == 'cnn_features':
        predictor.load_cnn_features(topN=50,
                                   dimensionality_reduction=False)
      elif fv_type == 'pred_features':
        predictor.make_feature_vectors()

    prediction_methods = {'avg_pool': predictor.avg_pool_prediction,
                        'max_pool': predictor.max_pool_prediction,
                        'svm': predictor.SVM_prediction, 
                        'knn': predictor.KNN_prediction,
                        'random_forest': predictor.random_forest_prediction,
                        }
  
    preds, labels = prediction_methods[prediction_method]()
    preds = preds['test'].reshape((-1, 1))
    labels = labels['test'].reshape((-1, 1))
    try:
      full_preds = np.concatenate((full_preds, preds), 1)
      full_labels = np.concatenate((full_labels, labels), 1)
    except NameError:
      full_preds = preds
      full_labels = labels
  
  avg_preds = np.mean(full_preds, 1)
  #labels = labels['test']
  metrics = evaluate_predictions(avg_preds, labels, threshold)
  return metrics



# ——————————————————————————————————————————————————————————————————————



  prediction_methods = {'avg_pool': predictor.avg_pool_prediction,
                        'max_pool': predictor.max_pool_prediction,
                        'svm': predictor.SVM_prediction, 
                        'knn': predictor.KNN_prediction,
                        'random_forest': predictor.random_forest_prediction,
                        }



     prediction_methods = {'avg_pool': predictor.avg_pool_prediction,
                          'max_pool': predictor.max_pool_prediction,}




   if prediction_method not in ['avg_pool', 'max_pool']:
    modes.pop('train')

  if prediction_method in ['svm','knn', 'random_forest']:
    if fv_type == 'cnn_features':
      predictor.load_cnn_features(topN=50,
                                   dimensionality_reduction=False)
    elif fv_type == 'pred_features':
      predictor.make_feature_vectors()
 
# ——————————————————————————————————————————————————————————————————————

def plot_histograms(subdir, case_split):
  """ Plots and saves histograms of prediction values
  """
  predictor = SlidePrediction(subdir, case_split)
  Path(histogram_dir).mkdir(parents=True, exist_ok=True)
  predictor.plot_histograms(histogram_dir)

