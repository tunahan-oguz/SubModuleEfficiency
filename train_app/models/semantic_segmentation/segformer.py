
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from train_app.registers.model_registry import model_registry
from train_app.models.base import SemanticSegmentationAdapter

@model_registry.register("SegFormerb0")
class SegFormerb0(SemanticSegmentationAdapter):
    def __init__(self, num_classes, target_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes
        )
        self.target_shape = target_shape


    def forward(self, x):
        outputs = self.model(x)
        return torch.nn.functional.interpolate(outputs.logits, self.target_shape)
    
@model_registry.register("SegFormerb2")
class SegFormerb2(SemanticSegmentationAdapter):
    def __init__(self, num_classes, target_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=num_classes
        )
        self.target_shape = target_shape


    def forward(self, x):
        outputs = self.model(x)
        return torch.nn.functional.interpolate(outputs.logits, self.target_shape)
    