import os
import sys
import time
import tqdm
import torch
import torchvision.transforms as transforms
import torch.onnx
import onnx
from onnxsim import simplify

import numpy as np

from PIL import Image

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

import faulthandler
faulthandler.enable()
from lib.InSPyReNet import InSPyReNet_SwinB
from lib.InSPyReNet import InSPyReNet_Res2Net50
from lib import *
from utils.misc import *

from data.dataloader import *
from data.custom_transforms import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
def _args():
   parser = argparse.ArgumentParser()
   parser = argparse.ArgumentParser()
   parser.add_argument('--config', '-c',     type=str,            default='configs/InSPyReNet_Res2Net50_change.yaml')
   parser.add_argument('--source', '-s',     type=str)
   parser.add_argument('--dest', '-d',       type=str,            default=None)
   parser.add_argument('--type', '-t',       type=str,            default='map')
   parser.add_argument('--gpu', '-g',        action='store_true', default=False)
   parser.add_argument('--jit', '-j',        action='store_true', default=False)
   parser.add_argument('--verbose', '-v',    action='store_true', default=False)
   return parser.parse_args()
def get_format(source):
    img_count = len([i for i in source if i.lower().endswith(('.jpg', '.png', '.jpeg'))])
    vid_count = len([i for i in source if i.lower().endswith(('.mp4', '.avi', '.mov' ))])
    
    if img_count * vid_count != 0:
        return ''
    elif img_count != 0:
        return 'Image'
    elif vid_count != 0:
        return 'Video'
    else:
        return ''

args = _args()
opt = load_config(args.config)
model = eval(opt.Model.name)(**opt.Model)


  ##loading the model 

model_weights="/media/rtiwari/raj/Inspyrenet_2/InSPyReNet/data/resnet_red_fur_2/latest.pth"

model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')), strict=True)
model.eval()
batch_size=3
args = _args()
opt = load_config(args.config)

if args.gpu is True:
    model = model.cuda()
model.eval()
x = torch.randn(batch_size, 3, 384, 384, requires_grad=True)
model=Simplify(model)
model = torch.jit.trace(model, torch.rand(1, 3, *opt.Test.Dataset.transforms.static_resize.size), strict=False)
output_file = "/media/rtiwari/raj/Inspyrenet_2/InSPyReNet/data/inspyrenet_latest.onnx"
torch.onnx.export(model,
                      x,
                      output_file,
                      opset_version=12,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}},
                      verbose=True)

import onnx

onnx_model = onnx.load(output_file)
onnx.checker.check_model(onnx_model)