import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

logging.getLogger(__name__)

class SegFormer(nn.Module):
    def __init__(self, model_name, in_channels, classes) -> None:
        super().__init__()
        
        if in_channels != 3:
            logging.critical(F"Segformer model expects three channels input")
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name,
                                                                      num_labels=classes,
                                                                      ignore_mismatched_sizes=True)

    def forward(self, img):
        output = self.model(img)
        logits = output.logits
        upsampled_logits = F.interpolate(input=logits, size=img.shape[2:], 
                                         scale_factor=None, mode='bilinear', align_corners=False)
        return upsampled_logits