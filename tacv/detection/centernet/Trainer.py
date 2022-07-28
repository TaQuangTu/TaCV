from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import yaml
import torch

from .backbones import get_backbone
from .CenterNet import CenterNet

TRAIN_CONFIG = "train_config"
VAL_CONFIG = "val_config"


class CenterNetBackboneFineTuning(BackboneFinetuning):
    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        print("Freeze layer before training")
        backbone = pl_module.backbone
        for name, param in backbone.named_parameters():
            if "deconv_layers" not in name:
                print(f"Freezing {name}")
                param.requires_grad = False

    def finetune_function(
            self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        print(f"Finetune function epoch {epoch}")
        if self.unfreeze_backbone_at_epoch == epoch:
            backbone = pl_module.backbone
            for name, param in backbone.named_parameters():
                if "deconv_layers" not in name:
                    print(f"Unfreezing {name}")
                    param.requires_grad = True


def create_checkpoint_callback(config):
    checkpoint_callback = ModelCheckpoint(
        monitor=config["monitor"],
        dirpath=config["dirpath"],
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}.pth",
        save_top_k=config["save_top_k"],
        mode=config["mode"],
        save_weights_only=True
    )
    return checkpoint_callback


def create_backbone_unfreeze_callback(config):
    unfreeze_bbone_at_epoch = config["unfreeze_bbone_epoch"]
    initial_denom_lr = config["initial_denom_lr"]
    callback = CenterNetBackboneFineTuning(
        unfreeze_bbone_at_epoch, initial_denom_lr=initial_denom_lr
    )
    return callback


def load_model_for_inference(config_path, device, load_pretrained_backbone=True):
    with open(config_path, "r") as config_file:
        yaml_config = yaml.safe_load(config_file)
        model = load_centernet_model_with_config(yaml_config, load_pretrained_backbone).to(device)
        ckpt_path = yaml_config["model"]["ckpt"]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def load_centernet_model_with_config(config, load_bbone_pretrained=True):
    layers = config["model"]["backbone_layers"]
    backbone = get_backbone(layers, load_bbone_pretrained)
    num_classes = config["model"]["num_classes"]
    head_conv_channel = int(config["model"]["head_conv_channel"])
    max_object = int(config["model"]["max_object"])
    input_shape = config["model"]["input_shape"]
    model = CenterNet(backbone, num_classes, head_conv_channel, max_object, input_shape, config[TRAIN_CONFIG])
    return model


class CenterNetTrainer:
    def __init__(self, train_data, val_data, config_path):
        self.train_data = train_data
        self.val_data = val_data
        self.config = yaml.safe_load(open(config_path, "r"))
        self.train_bs = self.config[TRAIN_CONFIG]["batch_size"]
        self.val_bs = self.config[VAL_CONFIG]["batch_size"]
        self.shuffle = self.config[TRAIN_CONFIG]["shuffle"]
        self.num_workers = self.config[TRAIN_CONFIG]["num_workers"]

    def train(self):
        # load model
        model = load_centernet_model_with_config(self.config)

        train_loader = DataLoader(self.train_data, self.train_bs, self.shuffle, num_workers=self.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(self.val_data, self.val_bs, num_workers=self.num_workers)

        # create callbacks
        checkpoint_callback = create_checkpoint_callback(self.config[TRAIN_CONFIG]["callback"])
        backbone_unfreeze_callback = create_backbone_unfreeze_callback(self.config[TRAIN_CONFIG])

        # config trainer
        num_gpus = self.config[TRAIN_CONFIG]["gpus"]
        gpus = None if num_gpus == 0 else num_gpus
        #
        trainer = Trainer(gpus=gpus, max_epochs=self.config["train_config"]["epoch"],
                          callbacks=[checkpoint_callback, backbone_unfreeze_callback], amp_backend="apex",
                          amp_level="02",
                          auto_lr_find=True)
        trainer.fit(model, train_loader, val_loader)
