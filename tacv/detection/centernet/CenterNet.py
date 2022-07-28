from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential, Conv2d, LeakyReLU, Sigmoid, ReLU
import numpy as np
from torch.nn.functional import binary_cross_entropy, mse_loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from .losses import centerness_loss
from .utils.data_utils import gaussian_radius, draw_umich_gaussian


class CenterNet(LightningModule):
    def __init__(self, backbone, num_classes, head_conv_channel=64, max_object=64, input_shape=(480, 640),
                 train_config=None):
        super(CenterNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.class_head = self.init_class_head(head_conv_channel)
        self.center_off_head = self.init_center_offset_head(head_conv_channel)
        self.regress_head = self.init_regression_head(head_conv_channel)
        self.max_object = max_object
        self.input_shape = input_shape
        self.down_ratio_x = 4
        self.down_ratio_y = 4
        self.calculate_hm_shape()
        self.train_config = train_config
        self.loss_hm_offset_offset_weights = [1, 1, 0.1]
        if train_config is not None:
            self.loss_hm_offset_offset_weights = train_config["loss_hm_offset_offset_weights"]

    def calculate_hm_shape(self):
        mock_input = torch.rand(1, 3, self.input_shape[0], self.input_shape[1])
        output = self(mock_input)
        hm = output["hm"]
        h, w = hm.shape[-2:]
        self.down_ratio_x = self.input_shape[1] / w
        self.down_ratio_y = self.input_shape[0] / h

    def init_class_head(self, head_conv_channel):
        return Sequential(
            Conv2d(64, head_conv_channel,
                   kernel_size=3, padding=1, bias=True),
            LeakyReLU(inplace=True),
            Conv2d(head_conv_channel, self.num_classes,
                   kernel_size=1, stride=1,
                   padding=0, bias=True),
            Sigmoid()
        )

    def init_regression_head(self, head_conv_channel):
        return Sequential(
            Conv2d(64, head_conv_channel,
                   kernel_size=3, padding=1, bias=True),
            LeakyReLU(inplace=True),
            Conv2d(head_conv_channel, 2,
                   kernel_size=1, stride=1,
                   padding=0, bias=True),
            ReLU(inplace=True)
        )

    def init_center_offset_head(self, head_conv_channel):
        return Sequential(
            Conv2d(64, head_conv_channel,
                   kernel_size=3, padding=1, bias=True),
            LeakyReLU(inplace=True),
            Conv2d(head_conv_channel, 2,
                   kernel_size=1, stride=1,
                   padding=0, bias=True),
            Sigmoid()
        )

    def forward(self, x):
        feature = self.backbone(x)
        heatmap = self.class_head(feature)
        offset = self.center_off_head(feature)
        regression = self.regress_head(feature)
        return {
            "hm": heatmap,
            "offset": offset,
            "reg": regression
        }

    def training_step(self, batch, batch_idx):
        image = batch["image"]  # N x 3 x H X W
        annos = batch["annos"]  # N x max_obj x 5
        masks = batch["masks"]  # N x max_obj
        masks = masks.unsqueeze(dim=-1)
        model_output = self(image)
        batch_size = image.shape[0]
        num_objs = torch.nonzero(masks).__len__()
        gt_hm, gt_reg = self.create_gt_heatmap(annos, masks, batch_size)
        gt_offset = gt_reg[:, :, :2]
        gt_wh = gt_reg[:, :, 2:]
        pred_hm, pred_reg = self.gather_output(annos, masks, batch_size, model_output)
        pred_offset = pred_reg[:, :, :2]
        pred_wh = pred_reg[:, :, 2:]
        loss_hm = centerness_loss(gt_hm, pred_hm) * self.loss_hm_offset_offset_weights[0]
        loss_offset = binary_cross_entropy(pred_offset, gt_offset, reduction="none") * masks / (num_objs + 0.0001)
        loss_wh = mse_loss(pred_wh, gt_wh, reduction="none") * masks / (num_objs + 0.0001)
        loss_offset = loss_offset.sum() * self.loss_hm_offset_offset_weights[1]
        loss_wh = loss_wh.sum() * self.loss_hm_offset_offset_weights[2]
        total = loss_hm + loss_offset + loss_wh
        print("Losses:", f"Train Loss hm: {loss_hm}, Loss offset: {loss_offset}, Loss wh: {loss_wh}")
        return total

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        image = val_batch["image"]  # N x 3 x H X W
        annos = val_batch["annos"]  # N x max_obj x 4
        masks = val_batch["masks"]  # N x max_obj
        masks = masks.unsqueeze(dim=-1)
        model_output = self(image)
        batch_size = image.shape[0]
        num_objs = torch.nonzero(masks).__len__()
        gt_hm, gt_reg = self.create_gt_heatmap(annos, masks, batch_size)
        gt_offset = gt_reg[:, :, :2]
        gt_wh = gt_reg[:, :, 2:]
        pred_hm, pred_reg = self.gather_output(annos, masks, batch_size, model_output)
        pred_offset = pred_reg[:, :, :2]
        pred_wh = pred_reg[:, :, 2:]
        loss_hm = centerness_loss(gt_hm, pred_hm) * self.loss_hm_offset_offset_weights[0]
        loss_offset = binary_cross_entropy(pred_offset, gt_offset, reduction="none") * masks / (num_objs + 0.0001)
        loss_wh = mse_loss(pred_wh, gt_wh, reduction="none") * masks / (num_objs + 0.0001)
        loss_offset = loss_offset.sum() * self.loss_hm_offset_offset_weights[1]
        loss_wh = loss_wh.sum() * self.loss_hm_offset_offset_weights[2]
        total = loss_hm + loss_offset + loss_wh
        print("Val:", f"Train Loss hm: {loss_hm}, Loss offset: {loss_offset}, Loss wh: {loss_wh}")
        self.log("val_loss", total)

    def backward(self, loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], *args,
                 **kwargs) -> None:
        loss.backward()

    def gather_output(self, annos, masks, batch_size, output):
        """
        :param annos:
        :param masks:
        :param batch_size:
        :param output: model output
        :return:
        """
        hm = output["hm"]
        reg = torch.zeros((batch_size, self.max_object, 4), device=self.device)  # 2 for offset and 2 for wh

        model_reg_output = output["reg"]
        model_offset_output = output["offset"]

        for image_index, (image_annos, obj_masks) in enumerate(zip(annos, masks)):
            for obj_index, (value, anno) in enumerate(zip(obj_masks, image_annos)):
                if value.item() is True:
                    x, y, w, h, class_id = anno
                    x, y, w, h = x.item(), y.item(), w.item(), h.item()
                    down_y = y / self.down_ratio_y
                    down_x = x / self.down_ratio_x

                    down_y_int = int(down_y)
                    down_x_int = int(down_x)
                    output_offset_x = model_offset_output[image_index, 0, down_y_int, down_x_int]
                    output_offset_y = model_offset_output[image_index, 1, down_y_int, down_x_int]
                    output_w = model_reg_output[image_index, 0, down_y_int, down_x_int]
                    output_h = model_reg_output[image_index, 1, down_y_int, down_x_int]

                    reg[image_index, obj_index, 0] = output_offset_x
                    reg[image_index, obj_index, 1] = output_offset_y
                    reg[image_index, obj_index, 2] = output_w
                    reg[image_index, obj_index, 3] = output_h
                else:
                    continue
        return hm, reg

    def create_gt_heatmap(self, annos, masks, batch_size):
        hm_x = self.input_shape[1] / self.down_ratio_x
        hm_y = self.input_shape[0] / self.down_ratio_y
        hm = np.zeros((batch_size, self.num_classes, int(hm_y), int(hm_x)))
        reg = torch.zeros((batch_size, self.max_object, 4), device=self.device)  # 2 for offset and 2 for wh
        for index, (image_annos, obj_masks) in enumerate(zip(annos, masks)):
            for obj_index, (value, anno) in enumerate(zip(obj_masks, image_annos)):
                if value.item():
                    x, y, w, h, class_id = anno
                    x, y, w, h = x.item(), y.item(), w.item(), h.item()
                    class_id = int(class_id)

                    down_y = y / self.down_ratio_y
                    down_x = x / self.down_ratio_x

                    down_y_int = int(down_y)
                    down_x_int = int(down_x)
                    # build hm
                    radius = gaussian_radius((h, w))
                    radius = max(0, int(radius))
                    hm[index, class_id, :, :] = draw_umich_gaussian(hm[index, class_id], (down_x_int, down_y_int),
                                                                    radius)
                    # build regression gt
                    offset_y = down_y - down_y_int
                    offset_x = down_x - down_x_int

                    reg[index, obj_index, 0] = offset_x
                    reg[index, obj_index, 1] = offset_y
                    reg[index, obj_index, 2] = w
                    reg[index, obj_index, 3] = h
                else:
                    continue
        hm = torch.as_tensor(hm, device=self.device)
        return hm, reg

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config["learning_rate"],
                                     weight_decay=self.train_config["weight_decay"])
        lr_scheduler = MultiStepLR(optimizer, milestones=self.train_config["lr_decay_milestones"],
                                   gamma=self.train_config["lr_decay_gamma"])
        return [optimizer], [lr_scheduler]
