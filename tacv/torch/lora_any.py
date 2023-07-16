from typing import Iterable, Union, OrderedDict

import loralib as lora
import torch
from loguru import logger
from pytorch_lightning import LightningModule
from torch import nn

import tacv.torch


class AnyLoRA(LightningModule):
    """This class is designed to not be used standalone. Use it as an extra parent class in a multiple inheriting class, for example:
    class MyClass(AnyTorchModel, AnyLoRA):
        def __init__(param):
            AnyTorchModel.__init__(param)
    Remember to make the __init__ function the same AnyTorchModel.__init__ function
    """

    def _init_lora_on_layer(self, layer_name: str, rank):
        layer_components = layer_name.split(".")
        parent_component = self
        child_component_name = layer_components[0]
        layer = self.__getattr__(child_component_name)
        if len(layer_components) > 1:
            for component in layer_components[1:]:
                parent_component = layer
                child_component_name = component
                if child_component_name.isdigit():
                    layer = layer[int(child_component_name)]
                else:
                    layer = layer.__getattr__(child_component_name)

        if isinstance(layer, nn.Linear):
            new_layer = lora.Linear(in_features=layer.in_features, out_features=layer.out_features, r=rank)
        elif isinstance(layer, nn.Embedding):
            new_layer = lora.Embedding(num_embeddings=layer.num_embeddings, embedding_dim=layer.embedding_dim,
                                       r=rank)
        elif isinstance(layer, nn.Conv2d):
            new_layer = lora.Conv2d(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                    kernel_size=layer.kernel_size if isinstance(layer.kernel_size, int) else
                                    layer.kernel_size[0], r=rank)
        else:
            logger.warning(
                f"Currently we only support initializing LoRA on layer types [Linear, Conv2d, Embedding]. But found layer name {layer_name} of type {type(layer)}")
            return
        new_layer.weight.data = layer.weight.data
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data
        parent_component.__setattr__(child_component_name, new_layer)

    def init_lora(self, rank: Union[int, Iterable[int]], layers_start_with: Union[Iterable[str], str] = None,
                  layers_end_with: Union[Iterable[str], str] = None, layers_contain: Union[Iterable[str], str] = None):
        """
        Init loRA weights for layers:
        @param rank: rank of the LoRA weights
        @param layers_start_with: Init LoRA weights on layers having names starting with specified values
        @param layers_end_with: Init LoRA weights on layers having names ending with specified values
        @param layers_contain: Init LoRA weights on layers having names containing specified values
        """
        assert rank > 0 and isinstance(rank, int)
        assert layers_start_with or layers_end_with or layers_contain

        if layers_start_with is None:
            layers_start_with = []
        if layers_end_with is None:
            layers_end_with = []
        if layers_contain is None:
            layers_contain = []

        if isinstance(layers_start_with, str):
            layers_start_with = [layers_start_with]
        if isinstance(layers_end_with, str):
            layers_end_with = [layers_end_with]
        if isinstance(layers_contain, str):
            layers_contain = [layers_contain]

        # get all model param names
        model_param_names = [name for name, _ in self.named_parameters()]
        unique_param_names = set()
        for p_name in model_param_names:
            splits = p_name.split(".")
            if len(splits) > 0:
                if splits[-1] == "weight":
                    unique_param_names.add(".".join(splits[:-1]))
                elif splits[-1] == "bias":
                    unique_param_names.add(".".join(splits[:-1]))
        model_param_names = list(unique_param_names)

        for p_name in model_param_names:
            for layer_contain in layers_contain:
                if layer_contain in p_name:
                    self._init_lora_on_layer(p_name, rank)

        for p_name in model_param_names:
            for layer_end_with in layers_end_with:
                if p_name.endswith(layer_end_with):
                    self._init_lora_on_layer(p_name, rank)

        for p_name in model_param_names:
            for layer_start_with in layers_start_with:
                if p_name.startswith(layer_start_with):
                    self._init_lora_on_layer(p_name, rank)

    def eval(self, break_lora=False):
        if not break_lora:
            logger.warning(
                "Fail to turn on eval mode. once the eval mode is on, we can not back to LoRA-training mode again, use break_lora=True to confirm")
            return
        return super(AnyLoRA, self).eval()

    def save_lora(self, path, bias="none"):
        lora_dict = lora.lora_state_dict(self, bias=bias)
        torch.save(lora_dict, path)

    def load_lora(self, pretrained: Union[str, OrderedDict], map_location="cpu"):
        tacv.torch.load_weights(self, pretrained, location=map_location)

    def mark_only_lora_as_trainable(self, bias="none"):
        lora.mark_only_lora_as_trainable(self, bias)
