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
import torch.nn.functional as F

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

class CustomMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


def Weighted_GAP(supp_feat, mask):
    #---
    print("mask type : ", type(mask))
    print("mask shape : ", mask.shape)
    mask = mask.unsqueeze(1)
    mask = mask.to(dtype=torch.float32)
    resized_mask = F.interpolate(mask, size=supp_feat.shape[-2:], mode='bilinear', align_corners=False)  # Shape: [2, 1, 112, 112]
    #---
    supp_feat = supp_feat * resized_mask
    #print("supp_feat : ", supp_feat)
    #feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    #print("feat_h, feat_w",feat_h, feat_w)
    #area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    
    #kernel_size = 3  # 3x3 kernel
    #stride = 1       # Stride of 1 to preserve size
    #padding = (kernel_size - 1) // 2  # Padding = 1 for 3x3 kernel
    
    #supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=kernel_size, stride=stride, padding=padding) * feat_h * feat_w / area
    
    return supp_feat

class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        #print("len(mlps)", len(mlps))
        self.linear_c = nn.ModuleList(mlps)
        
        #-- 
        self.custom_mlp = CustomMLP(input_dim=1024, output_dim=768)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        
        self.linear_fuse_support = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )        
        
        self.custom_linear_fuse = nn.Conv2d(
            in_channels=2304,
            out_channels=768,
            kernel_size=1,
            bias=False,
        )
        
        self.conv1x1 = nn.Conv2d(in_channels=112 * 112, out_channels=64, kernel_size=1)  # Reduce to 64 channels
        self.resnet_conv1x1 = nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=1)
        
        
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        self.custom_classifier = nn.Conv2d(768, config.num_labels, kernel_size=1)

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

        #----
        # support feat
        support_hidden_states = all_hidden_states[0:4]
        support_states = self.linear_fuse_support(torch.cat(support_hidden_states[::-1], dim=1)) 
        print("support_states ", support_states.shape)
        
        #----
        # query feat for dinov2
        # query_state = encoder_hidden_states[4]
        # print("query_states ", query_state.shape)
        # height, width = query_state.shape[2], query_state.shape[3]
        # query_state = self.custom_mlp(query_state)
        # query_state = query_state.permute(0, 2, 1)
        # query_state = query_state.reshape(batch_size, -1, height, width)
        # # upsample
        # query_state = nn.functional.interpolate(
        #     query_state, size=(112,112), mode="bilinear", align_corners=False
        # )       
        # print("query_states ", query_state.shape)
        
        #---
        # query feat for resnet 
        query_state = encoder_hidden_states[4]
        print("query_states ", query_state.shape)
        #print("Extracted features shape:", query_state.shape)  # Output: torch.Size([2, 2048])
        # Upsample spatial dimensions
        upsample = nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False)
        upsampled_spatial = upsample(query_state)  # [2, 2048, 112, 112]

        # Reduce channels using 1x1 convolution
        query_state = self.resnet_conv1x1(upsampled_spatial)  # [2, 768, 112, 112]

        #print(query_state.shape)  # Output: torch.Size([2, 768, 112, 112])
        
        
        #----------------------------------
        # Cosine similarity ! 
        # Normalize query and support states
        #query_norm = query_state / query_state.norm(dim=1, keepdim=True)  # Normalize along the channel dimension
        #support_norm = support_states / support_states.norm(dim=1, keepdim=True)
        #cosine_similarity = (query_norm * support_norm).sum(dim=1, keepdim=True)  # [2, 1, 112, 112]
        #print("cosine_similarity ", cosine_similarity.shape)  # Output: torch.Size([2, 1, 112, 112])

        # Spatial Correlation matrix
        # Reshape tensors to [batch, channels, height*width]
        #query_reshaped = query_state.view(2, 768, -1)  # [2, 768, 112*112]
        #support_reshaped = support_states.view(2, 768, -1)  # [2, 768, 112*112]

        # # Matrix multiplication for full correlation
        # full_correlation = torch.bmm(query_reshaped.permute(0, 2, 1), support_reshaped)  # [2, 112*112, 112*112]
        # full_correlation = full_correlation.view(2, 112, 112, 112, 112)  # Reshape to spatial dimensions
        # #print(full_correlation.shape)  # Output: torch.Size([2, 112, 112, 112, 112])
        # full_correlation_reshaped = full_correlation.permute(0, 3, 4, 1, 2).contiguous()  # [Batch, H_s, W_s, H_q, W_q]
        # full_correlation_reshaped = full_correlation_reshaped.view(2, 112 * 112, 112, 112)  # [Batch, Channels (H_s*W_s), H_q, W_q]

        # # Define a 1x1 Conv to reduce channels
        # # => wrong approach 
        # reduced_correlation = self.conv1x1(full_correlation_reshaped)  # [Batch, Reduced Channels, H_q, W_q]
        # print(reduced_correlation.shape)  # Output: [2, 64, 112, 112]
                                
        #----------------------------------
        # Weighted GAP
        # Masked Average Pooling
        mask_ = encoder_hidden_states[5]
        weighted_support_states = Weighted_GAP(supp_feat=support_states, mask=mask_)
        print("weighted_support_states", weighted_support_states.shape)
        
                
        all_factors = torch.cat([query_state, support_states, weighted_support_states][::-1], dim=1)
        print("all_factors : ", all_factors.shape)
        hidden_states = self.custom_linear_fuse(all_factors)
        #-----
        #hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        # logits = self.classifier(hidden_states)
        logits = self.custom_classifier(hidden_states)

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
        
        # support hidden states
        encoder_hidden_states = list(encoder_hidden_states)
        # query hidden states
        encoder_hidden_states.append(dino_features)
        # label for support mask pooling 
        encoder_hidden_states.append(labels)
    
        
        
        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                #loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                #loss = loss_fct(upsampled_logits, labels)
                
                #------------------------------------------------------- 
                # custom cross entropy or dice loss
                class_weights = torch.tensor([0.1, 1.0])
                class_weights = class_weights.to("cuda:1")
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                loss = criterion(upsampled_logits, labels)

                
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
        
        
#--


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
        correlation matrix added
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
        
        # support hidden states
        encoder_hidden_states = list(encoder_hidden_states)
        # query hidden states
        encoder_hidden_states.append(dino_features)
        # label for support mask pooling 
        encoder_hidden_states.append(labels)
    
        
        
        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                #loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                #loss = loss_fct(upsampled_logits, labels)
                
                #------------------------------------------------------- 
                # custom cross entropy or dice loss
                class_weights = torch.tensor([0.1, 1.0])
                class_weights = class_weights.to("cuda:1")
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                loss = criterion(upsampled_logits, labels)

                
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