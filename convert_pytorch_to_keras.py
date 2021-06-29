import os
import json
import sys

import numpy as np
import torch

import pth_models
import misc

import onnx
from onnx2keras import onnx_to_keras
import io
import logging
from onnx import optimizer



meso_main = '/projects/pathology_char/pathology_char_results/mesothelioma'

ckpt_path = os.path.join(meso_main, 
                         'results/4-11-20-10x_norm/run_1/ckpt_best_val.pth')
run_dir = os.path.dirname(ckpt_path)

def pytorch_to_keras(
    model, args, input_shapes=None,
    change_ordering=False, verbose=False, name_policy=None,
    use_optimizer=False, do_constant_folding=False):
    """
    By given PyTorch model convert layers with ONNX.
    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shapes: keras input shapes (using for each InputLayer)
        change_ordering: change CHW to HWC
        verbose: verbose output
        name_policy: use short names, use random-suffix or keep original names for keras layers
    Returns:
        model: created keras model.
    """
    logger = logging.getLogger('pytorch2keras')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info('Converter is called.')

    if name_policy:
        logger.warning('Name policy isn\'t supported now.')

    if input_shapes:
        logger.warning('Custom shapes isn\'t supported now.')

    if input_shapes and not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    if not isinstance(args, list):
        args = [args]

    args = tuple(args)

    dummy_output = model(*args)

    if isinstance(dummy_output, torch.autograd.Variable):
        dummy_output = [dummy_output]

    input_names = ['input_{0}'.format(i) for i in range(len(args))]
    output_names = ['output_{0}'.format(i) for i in range(len(dummy_output))]

    logger.debug('Input_names:')
    logger.debug(input_names)

    logger.debug('Output_names:')
    logger.debug(output_names)

    stream = io.BytesIO()
    torch.onnx.export(model, args, stream, 
                      do_constant_folding=do_constant_folding, 
                      verbose=verbose, input_names=input_names, 
                      output_names=output_names)

    stream.seek(0)
    onnx_model = onnx.load(stream)
    if use_optimizer:
        if use_optimizer is True:
            optimizer2run = optimizer.get_available_passes()
        else:
            use_optimizer = set(use_optimizer)
            optimizer2run = [x for x in optimizer.get_available_passes() 
                             if x in use_optimizer]
        logger.info("Running optimizer:\n%s", "\n".join(optimizer2run))
        onnx_model = optimizer.optimize(onnx_model, optimizer2run)

    k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names,
                            input_shapes=input_shapes, name_policy='renumerate',
                            verbose=verbose, change_ordering=change_ordering)

    return k_model


if __name__ == "__main__":

  # load pytorch model
  p_model = pth_models.initialize_model('resnet18', num_classes=2)
  state_dict = torch.load(ckpt_path, map_location='cpu')
  p_model.load_state_dict(state_dict)

  # dummy data
  input_np = np.random.uniform(0,1, (1, 3, 512, 512))
  input_var = (torch.FloatTensor(input_np))

  # convert to keras model
  tf_model = pytorch_to_keras(p_model, input_var, input_shapes=None,
                             change_ordering=False, verbose=False, 
                             name_policy=None,)
  #print(tf_model.summary())

  # save weights
  tf_model.save_weights(os.path.join(run_dir, 'model_weights.h5'))

  # conver to json
  model_string = tf_model.to_json()
  json_ = json.loads(model_string)
  layers = json_['config']['layers']
  
  # change configuration to load with straight keras
  for layer in layers[:]:
    layer['config'].pop('ragged', None)
    if layer['name'] in ['LAYER_66_EXPAND1', 'LAYER_66_EXPAND2',
                         'LAYER_67', 'LAYER_68_reshape']:
      layers.remove(layer)
    if layer['class_name'] == 'Conv2D':
      layer['config']['padding'] = 'same'
    if layer['class_name'] == 'ZeroPadding2D':
      layers.remove(layer)
  
  for i, layer in enumerate(layers):
    if i > 0:
      inbound = layer['inbound_nodes'][0][0][0]
      if 'pad' in inbound:
        layer['inbound_nodes'] = [[[layers[i-1]['name'], 0, 0, {}]]]
  
  for layer in layers:
    config = layer['config']
    for k in config.keys():
      if type(config[k]) == list:
        if len(config[k]) > 1:
          config[k] = tuple(config[k])
        else:
          config[k] = config[k][0]
      if type(config[k]) == tuple and len(config[k]) == 1:
        config[k] = config[k][0]
        
  layers[-1]['inbound_nodes'] = [[[layers[-2]['name'], 0, 0, {}]]]
  json_['keras_version'] = '2.2.4'

  # save new json config
  new_string = json.dumps(json_)
  with open(os.path.join(run_dir, 'keras_model.json'), 'w') as f:
    json.dump(new_string, f)

