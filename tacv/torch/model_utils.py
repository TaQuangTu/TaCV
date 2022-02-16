from typing import Union
import torch

def load_weights(model, source: Union[str,torch.nn.Module], name_matching_required=False, shape_matching_required=False, location="cpu", name_must_match: list = []):
    if isinstance(source,str):
        state_dict = torch.load(source, map_location=location)
    else:
        state_dict = source
    for name, param in model.named_parameters():
        if name not in state_dict:
            if name_matching_required:
                return False
            else:
                continue
        if param.shape != state_dict[name].shape:
            if shape_matching_required:
                return False
            else:
                continue
        if len(name_must_match)!=0:
            has_the_layers = False
            for specified_name in name_must_match:
                if specified_name in name:
                    has_the_layers = True
                    break
            if not has_the_layers:
                continue
        param.data.copy_(state_dict[name])
    return True
