import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.png_dataset import PNGDataset
from models.illumigan_model import IllumiganModel
from utils import Average, TrainManager
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True

import coremltools
import onnx;
from onnx_coreml import convert

base_dir = "_default/"
server = False

if len(sys.argv) > 1:
    base_dir = str(sys.argv[1])

if len(sys.argv) > 2:
    base_dir = str(sys.argv[1])
    server = True

# ------- 

options = './experiments/base_model/local_options.json'
hyperparams = './experiments/base_model/local_params.json'

if server:
    options = './experiments/base_model/options.json'
    hyperparams = './experiments/base_model/params.json'

# Function to mark the layer as output
# https://forums.developer.apple.com/thread/81571#241998
def convert_multiarray_output_to_image(spec, feature_name, is_bgr=False): 
    """ 
    Convert an output multiarray to be represented as an image 
    This will modify the Model_pb spec passed in. 
    Example: 
        model = coremltools.models.MLModel('MyNeuralNetwork.mlmodel') 
        spec = model.get_spec() 
        convert_multiarray_output_to_image(spec,'imageOutput',is_bgr=False) 
        newModel = coremltools.models.MLModel(spec) 
        newModel.save('MyNeuralNetworkWithImageOutput.mlmodel') 
    Parameters 
    ---------- 
    spec: Model_pb 
        The specification containing the output feature to convert 
    feature_name: str 
        The name of the multiarray output feature you want to convert 
    is_bgr: boolean 
        If multiarray has 3 channels, set to True for RGB pixel order or false for BGR 
    """
    for output in spec.description.output: 
        if output.name != feature_name: 
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType': 
            raise ValueError("%s is not a multiarray type" % output.name) 
        array_shape = tuple(output.type.multiArrayType.shape) 
        channels, height, width = array_shape 
        from coremltools.proto import FeatureTypes_pb2 as ft 
        if channels == 1: 
            output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE') 
        elif channels == 3: 
            if is_bgr: 
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR') 
            else: 
                output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB') 
        else: 
            raise ValueError("Channel Value %d not supported for image inputs" % channels) 
        output.type.imageType.width = width 
        output.type.imageType.height = height 
 

# working code
manager = TrainManager(base_dir=base_dir,
                       options_f_dir=options,
                       hyperparams_f_dir=hyperparams)

#dataset = JPGDataset(manager, 'short', 'long', transforms=True)
dataset = PNGDataset(manager, 'in', 'out', transforms=True)
dataloader = DataLoader(dataset, batch_size=manager.get_hyperparams().get(
    'batch_size'), shuffle=True, num_workers=0)

model = IllumiganModel(manager=manager)

dummy_input = torch.randn(1, 3, 1024, 1024).to(manager.device)

torch.onnx.export(model.generator_net, dummy_input, "Illumigan.onnx")

onnx_model = onnx.load('./Illumigan.onnx')

# If we normalize between -1 and 1 
# use this scale 
#scale = 2/255.0
#args = dict(
#    is_bgr=False,
#    red_bias = -1,
#    green_bias = -1, 
#    blue_bias = -1,
#    image_scale = scale
#)

# IF we normalize between 0 and 1 and input
# use this scale
scale = 1/255.0
args = dict(
    is_bgr=False,
    red_bias = 0,
    green_bias = 0, 
    blue_bias = 0,
    image_scale = scale
)
mlmodel = convert(onnx_model, image_input_names='0', preprocessing_args=args, image_output_names=['133']) # This is what makes it an image lol 
mlmodel.save('Illumigan.mlmodel')

coreml_model = coremltools.models.MLModel('Illumigan.mlmodel')
spec = coreml_model.get_spec()
spec_layers = getattr(spec,spec.WhichOneof("Type")).layers

# find the current output layer and save it for later reference
last_layer = spec_layers[-1]
 
# add the post-processing layer
new_layer = spec_layers.add()
new_layer.name = 'convert_to_image'
 
# Configure it as an activation layer
new_layer.activation.linear.alpha = 255
new_layer.activation.linear.beta = 0
 
# Use the original model's output as input to this layer
new_layer.input.append(last_layer.output[0])
 
# Name the output for later reference when saving the model
new_layer.output.append('image_output')
 
# Find the original model's output description
output_description = next(x for x in spec.description.output if x.name==last_layer.output[0])
 
# Update it to use the new layer as output
output_description.name = new_layer.name


# Mark the new layer as image
convert_multiarray_output_to_image(spec, output_description.name, is_bgr=False)

updated_model = coremltools.models.MLModel(spec)
 
updated_model.author = 'Magnus'
updated_model.license = 'None'
updated_model.short_description = 'Illumigan'
updated_model.input_description['image'] = 'Input Image'
updated_model.output_description[output_description.name] = 'Predicted Image'
 
model_file_name = 'Illumigan2.mlmodel'
updated_model.save(model_file_name)

