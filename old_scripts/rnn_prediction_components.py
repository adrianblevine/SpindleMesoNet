


def eval_single_run(ic_idx, config, folders):
  # only run on one of the cross validation runs
  # get split correspondign to cv run or whole split dict
  subdir = os.path.join(folders.run_dir, 'run_{}'.format(FLAGS.cv_idx))
  run_folders = get_run_folders(folders, subdir)
  run_folders.test_subdirs = folders.test_subdirs

  run_split = config.case_split.get(str(FLAGS.cv_idx), config.case_split) 
  split = {'train': run_split['train'],
            'val': run_split['test'] }
 
  preds, labels, _ = predict_run(run_folders.subdir, split, 
                              run_folders.cnn_pred, 
                              config.prediction_type, config.fv_length, 
                              config.n_process) 
  if evaluate:
    evaluate_run(preds, labels, threshold)
  
  if write_to_csv:
    write_to_csv_file(csv_file, reset_csv)



#          if save_incorrect:
#            pred = binarize_predictions(pred, threshold=0.5)
#            for idx in range(len(pred)):
#              if pred[idx] != label[idx]:
#                try:
#                  id = os.path.splitext(ids[idx])[0]
#                  img_subdir = ('tumor_full' if id.startswith('MM_sarc')
#                                else 'benign')   
#                  img_path = os.path.join(region_img_dir, img_subdir, 
#                                          id + '.png')
#                  save_path = os.path.join(img_save_dir, id + '.jpg')
#                  img = Image.open(img_path)
#                  img.save(save_path)
#                  n_saved += 1
#                except FileNotFoundError as e:
#                  print(ids[idx], e) 

