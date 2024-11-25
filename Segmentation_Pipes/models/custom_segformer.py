from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn
import math

from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.segformer.configuration_segformer import SegformerConfig

# class SegformerPreTrainedModel(PreTrainedModel):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = SegformerConfig
#     base_model_prefix = "segformer"
#     main_input_name = "pixel_values"

#     def _init_weights(self, module):
#         """Initialize the weights"""
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)



# #---------------------------------------------------------
# class SegformerMLP(nn.Module):
#     """
#     Linear Embedding.
#     """

#     def __init__(self, config: SegformerConfig, input_dim):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

#     def forward(self, hidden_states: torch.Tensor):
#         hidden_states = hidden_states.flatten(2).transpose(1, 2)
#         hidden_states = self.proj(hidden_states)
#         return hidden_states


# class SegformerDecodeHead(SegformerPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
#         mlps = []
#         for i in range(config.num_encoder_blocks):
#             mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
#             mlps.append(mlp)
#         self.linear_c = nn.ModuleList(mlps)

#         # the following 3 layers implement the ConvModule of the original implementation
#         self.linear_fuse = nn.Conv2d(
#             in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
#             out_channels=config.decoder_hidden_size,
#             kernel_size=1,
#             bias=False,
#         )
#         self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
#         self.activation = nn.ReLU()

#         self.dropout = nn.Dropout(config.classifier_dropout_prob)
#         self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

#         self.config = config

#     def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
#         batch_size = encoder_hidden_states[-1].shape[0]

#         all_hidden_states = ()
#         for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
#             if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
#                 height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
#                 encoder_hidden_state = (
#                     encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
#                 )

#             # unify channel dimension
#             height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
#             encoder_hidden_state = mlp(encoder_hidden_state)
#             encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
#             encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
#             # upsample
#             encoder_hidden_state = nn.functional.interpolate(
#                 encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
#             )
#             all_hidden_states += (encoder_hidden_state,)

#         hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
#         hidden_states = self.batch_norm(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         # logits are of shape (batch_size, num_labels, height/4, width/4)
#         logits = self.classifier(hidden_states)

#         return logits
    


#---
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.segformer.modeling_segformer import SegformerPreTrainedModel
from transformers.models.segformer.modeling_segformer import SegformerModel
#from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional, Tuple, Union


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits


class CustomSegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
    
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits

class FewShotFormer(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        dino_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]


        #------------------------- 
        # DECODER
        
        encoder_hidden_states = list(encoder_hidden_states)
        encoder_hidden_states.append(dino_features)
        
        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )