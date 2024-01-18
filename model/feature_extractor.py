## Ref. https://github.com/amazon-science/patchcore-inspection

import copy
import torch
import torchvision.models as models
import torchvision.transforms as T

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            pass
        return None


class FeatureExtractor(torch.nn.Module):
    def __init__(self, device, backbone=None, extract_layer_names=('layer2', 'layer3')):
        super(FeatureExtractor, self).__init__()
        self.device = device
        if backbone is None:
            self.backbone = models.wide_resnet50_2(pretrained=True)
        else:
            self.backbone = backbone
        self.transform = T.Compose([T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

        self.backbone.hook_handles = []
        self.outputs = {}
        self.extract_layers = []
        for extract_layer_name in extract_layer_names:
            forward_hook = ForwardHook(self.outputs, extract_layer_name, extract_layer_names[-1])
            self.backbone.hook_handles.append(self.backbone.__dict__["_modules"][extract_layer_name][-1].register_forward_hook(forward_hook))
        self.pool_ratio = 8
        self.to(self.device)

    def forward(self, image):
        self.outputs.clear()

        with torch.no_grad():
            image = self.transform(image)
            self.backbone(image.unsqueeze(0).to(self.device))
        return self.outputs