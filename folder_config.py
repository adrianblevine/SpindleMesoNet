folder_config = {
  'train': {
    'image_subdirs': {'benign': 'benign', 'tumor': 'tumor_full'},
    'list': '',
    'output_dir': 'predictions_cv',
  },

  'val': {
    'image_subdirs': {'benign': 'benign', 'tumor': 'tumor_full'},
    'list': '',
    'output_dir': 'predictions_cv',
  },


  'referrals': {
    'image_subdirs':  {'benign': 'benign', 'tumor': 'tumor_full'},
    'list': '/path/to/dir/lists/pathologist_test_cases.txt',
    'output_dir': 'predictions_test',
  },

  'external': {
    'image_subdirs':  {'benign': 'ext_benign', 'tumor': 'ext_meso'},
    'list': '/path/to/dir/lists/external_test_set.txt',
    'output_dir': 'predictions_ext',
  }
}
