import os
import argparse
import random 
import numpy as np
import sys
import json
import copy

from PIL import Image
import matplotlib as mpl
from matplotlib import pyplot as plt

f = open(os.devnull, 'w')
sys.stderr = f; sys.stdout = f
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL) 
import keras
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
import innvestigate_utils as eutils
from innvestigate.applications import imagenet
sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__

from slide_prediction import get_label
import misc

#————————————————————————————————————————————————————————————————————

meso_main = '/projects/pathology_char/pathology_char_results/mesothelioma'

string_path = os.path.join(meso_main, 
                         'results/4-11-20-10x_norm/run_1/keras_model.json')

weights_path = os.path.join(meso_main, 
                         'results/4-11-20-10x_norm/run_1/model_weights.h5')

split_path = os.path.join(meso_main,
                          'results/4-11-20-10x_norm/img_split.pkl')

output_dir = os.path.join(meso_main, 'innvestigate_results')





#————————————————————————————————————————————————————————————————————
# from https://github.com/albermax/innvestigate/blob/master/examples/utils_imagenet.py

def preprocess(X, net):
  X = X.copy()
  X = net["preprocess_f"](X)
  return X

def postprocess(X, color_conversion, channels_first):
  X = X.copy()
  X = iutils.postprocess_images(
      X, color_coding=color_conversion, channels_first=channels_first)
  return X

def image(X):
  X = X.copy()
  return ivis.project(X, absmax=255.0, input_is_positive_only=True)

def bk_proj(X):
  X = ivis.clip_quantile(X, 1)
  return ivis.project(X)

def heatmap(X):
    #X = ivis.gamma(X, minamp=0, gamma=0.95)
  return ivis.heatmap(X)

def graymap(X):
  return ivis.graymap(np.abs(X), input_is_positive_only=True)

#————————————————————————————————————————————————————————————————————


def compare_methods(methods, model, images):
  analyzers = []
  for method in methods:
    try:
      analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                              model, # model without softmax output
                                              **method[1])      # optional analysis parameters
    except innvestigate.NotAnalyzeableModelException:
        # Not all methods work with all models.
      analyzer = None
    analyzers.append(analyzer)

  analysis = np.zeros([len(images), len(analyzers)]+[512, 512]+[3])
  text = []

  for i, (x, y) in enumerate(images):
    # Add batch axis.
    #x = x[None, :, :, :]
    x_pp = preprocess_numpy_input(x)

    # Predict final activations, probabilites, and label.
    prob = model.predict_on_batch(x_pp)[0]
    
    # Save prediction info:
    text.append(("%s" % y,    # ground truth label
                 "%.2f" % prob.max(),              # probabilistic softmax output  
                 ))

    for aidx, analyzer in enumerate(analyzers):
      print(methods[aidx][0])
      if methods[aidx][0] == "input":
        # Do not analyze, but keep not preprocessed input.
        vis = x/255
        plt.imshow(vis); plt.show()
      elif analyzer:
        # Analyze.
        vis = analyzer.analyze(x_pp)
        #import pdb;pdb.set_trace()
        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
        vis = postprocess(vis, None, channels_first=False)
        # Apply analysis postprocessing, e.g., creating a heatmap.
        vis = methods[aidx][2](vis)
      else:
        vis = np.zeros_like(image)
      analysis[i, aidx] = vis[0]
  
# Prepare the grid as rectangular list
  grid = [[analysis[i, j] for j in range(analysis.shape[1])]
          for i in range(analysis.shape[0])] 
             
# Prepare the labels
  label, prob, = zip(*text)
  row_labels_left = [('label: {}'.format(label[i])) for i in range(len(label))]
  row_labels_right = [('prob: {}'.format(prob[i])) for i in range(len(label))]
  col_labels = [''.join(method[3]) for method in methods]

# Plot the analysis.
  eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
              file_name=os.environ.get("plot_file_name", None))



def preprocess_numpy_input(x, mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225], 
                           data_format="channels_first"):
  # image is loaded with channels last, so need to change to channels
  # first to match the data_format parameter
  x = x.copy()
  x = np.rollaxis(x, 2 ,0)
  x /= 255.
  if data_format == 'channels_first':
    if x.ndim == 3:
      x[0, :, :] -= mean[0]
      x[1, :, :] -= mean[1]
      x[2, :, :] -= mean[2]
      if std is not None:
        x[0, :, :] /= std[0]
        x[1, :, :] /= std[1]
        x[2, :, :] /= std[2]
    else:
      x[:, 0, :, :] -= mean[0]
      x[:, 1, :, :] -= mean[1]
      x[:, 2, :, :] -= mean[2]
      if std is not None:
        x[:, 0, :, :] /= std[0]
        x[:, 1, :, :] /= std[1]
        x[:, 2, :, :] /= std[2]
  else:
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
      x[..., 0] /= std[0]
      x[..., 1] /= std[1]
      x[..., 2] /= std[2]
  if x.ndim == 3:
    x = np.expand_dims(x, 0) 
  return x


def load_random_image(img_list):
  img_path = random.choice(img_list)
  print(img_path)
  img = np.array(Image.open(img_path)).astype('float32')
  return (img, img_path)
 

def innvestigate_image(images, model, analysis_type):
  n_imgs = len(images)
  #plt.rcParams['figure.constrained_layout.use'] = True
  #fig, a = plt.figure(figsize=(8, 4 * n_imgs))
  fig, _axs = plt.subplots(n_imgs, 2, figsize=(9, 4*n_imgs),
                        subplot_kw={'xticks': [], 'yticks': []})
  axs = _axs.flatten()

  for i, data in enumerate(images):
    img = data[0]
    path = data[1]
    label = get_label(os.path.basename(path))
    print(label)
    original_img = copy.deepcopy(img)
    processed_img = preprocess_numpy_input(img)
    analyzer = innvestigate.create_analyzer(analysis_type, model)
    vis = analyzer.analyze(processed_img)
    # Aggregate along color channels and normalize to [-1, 1]
    vis = vis.sum(axis=np.argmax(np.asarray(vis.shape) == 3))
    vis += vis.min()
    vis /= np.max(np.abs(vis))
    
    # Plot
    #ax = fig.add_subplot(n_imgs, 2, 2*i +1)
    axs[2*i].imshow(original_img.astype('uint8'))
    axs[2*i].set_ylabel(label)
    #fig.add_subplot(n_imgs, 2, 2*i +2)
    # TODO: figure out best color map for each analyzer
    axs[2*i+1].imshow(vis[0], cmap="viridis", clim=(0, 1));
    #plt.imshow(a[0], cmap="seismic", clim=(-1, 1)); plt.axis('off')
  #plt.tight_layout()
  #import pdb; pdb.set_trace()
  #mappable = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1) , cmap='viridis') 
  plt.colorbar(plt.imshow(vis[0]), ax=axs, shrink=0.2, orientation='vertical')
  #fig.colorbar(mpl.cm.ScalarMappable(norm=vis[0], cmap='viridis'), ax=[1,3,5], shrink=0.6)
  #fig.align_labels() 
  plt.show()

#————————————————————————————————————————————————————————————————————

if __name__ == "__main__":
  if len(sys.argv) > 1:
    analysis_method = sys.argv[1]
  else:
    analysis_method = "deep_taylor"
  data_split = misc.load_pkl(split_path)['1']

  #————————————————————————————————————————————————————————————————————
  # set up method comparison
  #tmp = getattr(imagenet, 'densenet121')
  #net = tmp(load_weights=True, load_patterns="relu")
  patterns = None #net["patterns"]
  input_range = (-3, 3) #net["input_range"]

  noise_scale = (input_range[1]-input_range[0]) * 0.1

  methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN                TITLE
    # Show input.
    ("input",                 {},                       image,         "Input"),

    # Function
    ("gradient",              {"postprocess": "abs"},   graymap,       "Gradient"),
    #("smoothgrad",            {"augment_by_n": 64,
    #                           "noise_scale": noise_scale,
    #                           "postprocess": "square"},graymap,       "SmoothGrad"),

    # Signal
    ("deconvnet",             {},                       bk_proj,       "Deconvnet"),
    ("guided_backprop",       {},                       bk_proj,       "Guided Backprop",),
    #("pattern.net",           {"patterns": patterns},   bk_proj,       "PatternNet"),

    # Interaction
    #("pattern.attribution",   {"patterns": patterns},   heatmap,       "PatternAttribution"),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, heatmap,       "DeepTaylor"),
    ("input_t_gradient",      {},                       heatmap,       "Input * Gradient"),
    #("integrated_gradients",  {"reference_inputs": input_range[0],
    #                           "steps": 64},            heatmap,       "Integrated Gradients"),
    ("lrp.z",                 {},                       heatmap,       "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           heatmap,       "LRP-Epsilon"),
    ("lrp.sequential_preset_a_flat",{"epsilon": 1},     heatmap,       "LRP-PresetAFlat"),
    ("lrp.sequential_preset_b_flat",{"epsilon": 1},     heatmap,       "LRP-PresetBFlat"),
  ]
 
  #————————————————————————————————————————————————————————————————————
  # load keras model
  with open(os.path.join(string_path), 'r') as f:
    model_string = json.load(f)
  model = keras.models.model_from_json(model_string) 
  model.load_weights(weights_path)


  # Innvestigate
  if sys.argv[1] == 'compare_methods':
    # load sample image
    image, path = load_random_image(data_split['test'])
    images.append((image, get_label(path))) 
    compare_methods(methods, model, images)
  else:
    images = []
    for i in range(4):
      data = load_random_image(data_split['test'])
      images.append(data)
    innvestigate_image(images, model, analysis_method)
  
  sys.exit(0)

