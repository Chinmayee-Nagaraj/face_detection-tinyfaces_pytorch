"""
Script to detect the faces using tinyface
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from tinyfaces.evaluation import get_detections, get_model

def run(model, image, templates, prob_thresh, nms_thresh, device):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # Convert to tensor
    img = transforms.functional.to_tensor(image)

    rf = {'size': [859, 859], 'stride': [8, 8], 'offset': [-1, -1]}

    dets = get_detections(model,
                          img,
                          templates,
                          rf,
                          img_transforms,
                          prob_thresh,
                          nms_thresh,
                          scales=(0, ),
                          device=device)
    return dets


def main_fn(image, checkpoint, prob_thresh, nms_thresh):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    templates = json.load(open('tinyfaces/datasets/templates.json'))
    templates = np.round(np.array(templates), decimals=8)

    num_templates = templates.shape[0]

    model = get_model(checkpoint, num_templates=num_templates)

    with torch.no_grad():
        #run model on image
        dets = run(model, image, templates, prob_thresh, nms_thresh,
                   device)
    return dets

  
