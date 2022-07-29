import numpy as np
from torch import sigmoid
from torch.nn import MaxPool2d
import torch
import cv2

from .. import CenterNet


def decode_centernet(ct_hm, ct_offset_map, reg_map):
    bs, cn, hw, hh = ct_hm.shape

    max_pooling = MaxPool2d(kernel_size=3, stride=1, padding=1)

    max_per_channels = ct_hm == ct_hm.view(bs, cn, -1).max(dim=2, keepdim=True).values.unsqueeze(-1)
    positives = (ct_hm == max_pooling(ct_hm)) * (ct_hm >= 0.3) * (
            ct_hm == ct_hm.max(dim=1, keepdim=True).values.repeat(1, ct_hm.shape[1], 1, 1)) * max_per_channels
    pos_locs = positives.nonzero()

    batch_indices = pos_locs[:, 0:1]
    class_ids = pos_locs[:, 1]
    ct_offset = ct_offset_map[pos_locs[:, 0], :, pos_locs[:, 2], pos_locs[:, 3]]
    score = ct_hm[pos_locs[:, 0], pos_locs[:, 1], pos_locs[:, 2], pos_locs[:, 3]].unsqueeze(dim=1)
    wh = reg_map[pos_locs[:, 0], :, pos_locs[:, 2], pos_locs[:, 3]]
    return torch.cat((batch_indices, class_ids.unsqueeze(dim=1), pos_locs[:, 2].unsqueeze(dim=1),
                      pos_locs[:, 3].unsqueeze(dim=1), ct_offset, wh, score), dim=1)


def parse_output_centernet(pred):
    ct_hm = pred["hm"]
    offset = sigmoid(pred["offset"])
    reg = pred["reg"]
    boxes = decode_centernet(ct_hm, offset, reg)
    return boxes


@torch.no_grad()
def infer(model: CenterNet, image: np.ndarray, device="cpu"):
    h, w, c = image.shape()
    ih, iw = model.input_shape
    down_scale_h = h / ih
    down_scale_w = w / iw
    image = cv2.resize(image, dsize=(iw, ih), interpolation=cv2.INTER_LINEAR)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).to(device)
    image = torch.unsqueeze(image, 0)
    output = model(image)
    batch_output = parse_output_centernet(output)

    res = []
    for detection in batch_output:
        image_index, class_id, yint, xint, offsetx, offsety, w, h, score = detection
        class_id = int(class_id)
        yint = int(yint)
        xint = int(xint)
        offsetx = float(offsetx)
        offsety = float(offsety)
        w = float(w) * down_scale_w
        h = float(h) * down_scale_h
        score = round(float(score), 2)
        xc = (xint + offsetx) * down_scale_w * model.down_ratio_x
        yc = (yint + offsety) * down_scale_h * model.down_ratio_y
        res.append([xc, yc, w, h, class_id, score])
    return res
