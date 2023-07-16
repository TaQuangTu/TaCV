from typing import Optional, Tuple, Dict, Any, Union, List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import GPTNeoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .lora_any import AnyLoRA


# Copy from https://github.com/Lightning-AI/lit-gpt/blob/f50e2d8d6903863d6f558b7a1953bb4125ee5f96/lit_gpt/utils.py#L437
# this is to avoid OOM when compute gradient by splitting logits to smaller chunks and compute loss one by one
def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        return torch.cat(loss_chunks).mean()

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    return torch.cat(loss_chunks).mean()

class GPTNeoLoRA(GPTNeoForCausalLM, AnyLoRA):
    def __init__(self, config):
        GPTNeoForCausalLM.__init__(self, config)
        self.lm_head_chunk_size = 128

    # Copy and modified from transformers.GPTNeoForCausalLM
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            head_chunk_size=-1,
    ):
        r"""
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                    `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                    are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        if head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            lm_logits = [self.lm_head(x_i) for x_i in hidden_states.split(head_chunk_size, dim=1)]
        else:
            lm_logits = self.lm_head(hidden_states)

        loss = None
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def configure_optimizers(self):
        self.mark_only_lora_as_trainable(bias="none")
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters: {num_params}")
        num_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Number of non trainable parameters: {num_params}")

        optimizer = torch.optim.AdamW(trainable_params, lr=0.0002)
        scheduler = MultiStepLR(optimizer=optimizer, milestones=(45, 90, 180), gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_id):
        input_ids = batch["x"]
        targets = batch["y"]
        logits = self.forward(input_ids, return_dict=False, head_chunk_size=self.lm_head_chunk_size)[0]

        # shift the targets such that output n predicts token n+1
        logits[-1] = logits[-1][..., :-1, :]
        loss = chunked_cross_entropy(logits, targets[..., 1:])
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["x"]
        targets = batch["y"]
        logits = self.forward(input_ids, return_dict=False, head_chunk_size=self.lm_head_chunk_size)[0]
        # shift the targets such that output n predicts token n+1
        logits[-1] = logits[-1][..., :-1, :]
        loss = chunked_cross_entropy(logits, targets[..., 1:])
        self.log("val_loss", loss.item())
        return loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict = checkpoint["state_dict"]
        state_dict = {key: val for key, val in state_dict.items() if "lora_" in key}
        checkpoint["state_dict"] = state_dict


# TODO: Define your data loaders
train_data = YourDataset(data_dir / "train.pt")
val_data = YourDataset(data_dir / "test.pt")
val_data.max_seq_length = train_data.max_seq_length

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

model = GPTNeoLoRA.from_pretrained("EleutherAI/gpt-neo-1.3B", low_cpu_mem_usage=True)
model.init_lora(rank=8, layers_contain=["k_proj", "v_proj"])
model.mark_only_lora_as_trainable(bias="none")

# training callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10)
model_ckpt = ModelCheckpoint(dirpath="log_dir", filename='lora-epoch{epoch:02d}-val_loss{val_loss:.5f}',
                             monitor="val_loss", save_top_k=4)
logger = MLFlowLogger(experiment_name="lora_test")

trainer = Trainer(accelerator="gpu", devices=2, precision="16-mixed", callbacks=[early_stop, model_ckpt],
                  logger=logger, max_epochs=5000)
trainer.fit(model, train_loader, val_loader)

# Save the final LoRA checkpoint at the end of training
model.save_lora("lora.pth")