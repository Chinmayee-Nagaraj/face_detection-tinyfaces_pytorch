
import numpy as np
import torch
import torchvision
from torchvision import transforms

from tinyfaces.models.model import DetectionModel
from tinyfaces.models.utils import get_bboxes


def get_model(checkpoint=None, num_templates=25):
    model = DetectionModel(num_templates=num_templates)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
    return model


def get_detections(model,
                   img,
                   templates,
                   rf,
                   img_transforms,
                   prob_thresh=0.65,
                   nms_thresh=0.3,
                   scales=(-2, -1, 0, 1),
                   device=None):
    model = model.to(device)
    model.eval()

    dets = np.empty((0, 5))  # store bbox (x1, y1, x2, y2), score

    num_templates = templates.shape[0]

    # Evaluate over multiple scale
    scales_list = [2**x for x in scales]

    # convert tensor to PIL image so we can perform resizing
    image = transforms.functional.to_pil_image(img)

    min_side = np.min(image.size)

    for scale in scales_list:
        # scale the images
        scaled_image = transforms.functional.resize(image,
                                                    int(min_side * scale))

        # normalize the images
        img = img_transforms(scaled_image)

        # add batch dimension
        img.unsqueeze_(0)

        # now run the model
        x = img.float().to(device)

        output = model(x)

        # first `num_templates` channels are class maps
        score_cls = output[:, :num_templates, :, :]
        prob_cls = torch.sigmoid(score_cls)

        score_cls = score_cls.data.cpu().numpy().transpose((0, 2, 3, 1))
        prob_cls = prob_cls.data.cpu().numpy().transpose((0, 2, 3, 1))

        score_reg = output[:, num_templates:, :, :]
        score_reg = score_reg.data.cpu().numpy().transpose((0, 2, 3, 1))

        t_bboxes, scores = get_bboxes(score_cls, score_reg, prob_cls,
                                      templates, prob_thresh, rf, scale)

        scales = np.ones((t_bboxes.shape[0], 1)) / scale

        # append scores at the end for NMS
        d = np.hstack((t_bboxes, scores))

        dets = np.vstack((dets, d)) 

    scores = torch.from_numpy(dets[:, 4]) #tensor containing scores
    detts = torch.from_numpy(dets[:, :4]) #tensor including the boxes values, #changed dets to detts

    # Apply NMS
    keep = torchvision.ops.nms(detts, scores, nms_thresh)
    keep_np = keep.numpy()
   

    return dets[keep_np] #changes
