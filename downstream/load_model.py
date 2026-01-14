import os

os.environ['GITHUB_PROXY_URL'] = 'https://ghproxy.com/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from huggingface_hub import login

import timm


def extract_patches(x, patch_size, patch_resize, stride):

    patches = []

    batch_size, c, h, w = x.shape
    for i in list(set(list(range(0, h - patch_size + 1, stride)) + [h - patch_size])):
        for j in list(set(list(range(0, w - patch_size + 1, stride)) + [w - patch_size])):
            patch = x[:, :, i:i+patch_size, j:j+patch_size]
            new_patch = F.interpolate(patch, size=(patch_resize, patch_resize), mode='bilinear', align_corners=False)
            patches.append(new_patch)

    return patches



class ViT_classifier(nn.Module):

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):

        from model.vision_transformer import vit_large

        super(ViT_classifier, self).__init__()
        self.config = OmegaConf.load('config/custom_test.yaml')
        self.backbone = vit_large(
            patch_size=self.config.student.patch_size,
            init_values=self.config.student.layerscale,
            ffn_layer=self.config.student.ffn_layer,
            block_chunks=self.config.student.block_chunks,
            qkv_bias=self.config.student.qkv_bias,
            proj_bias=self.config.student.proj_bias,
            ffn_bias=self.config.student.ffn_bias,
            num_register_tokens=self.config.student.num_register_tokens,
            interpolate_offset=self.config.student.interpolate_offset,
            interpolate_antialias=self.config.student.interpolate_antialias,
        )

        self.classifier = nn.Linear(in_features=1024, out_features=num_class, bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x)  # x_norm_clstoken
        output = self.classifier(token)
        return output


class MIL_ViT_classifier(ViT_classifier):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ViT_classifier, self).__init__(pretrained_path=pretrained_path, num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output



class ViT_classifier_DINOV2(nn.Module):

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):


        super(ViT_classifier_DINOV2, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True)

        self.classifier = nn.Linear(in_features=1024, out_features=num_class, bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x)  # x_norm_clstoken
        output = self.classifier(token)
        return output



class MIL_ViT_classifier_DINOV2(ViT_classifier_DINOV2):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ViT_classifier_DINOV2, self).__init__(num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output



class ResNet50_classifier(nn.Module):

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):

        import torchvision.models as models

        super(ResNet50_classifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)    # ImageNet1k
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(in_features=2048, out_features=num_class, bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x)  # x_norm_clstoken
        output = self.classifier(token)
        return output



class MIL_ResNet50_classifier(ResNet50_classifier):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ResNet50_classifier, self).__init__(num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output



class ViT_classifier_UNI(nn.Module):

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):

        super(ViT_classifier_UNI, self).__init__()
        login(token="")
        self.backbone = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

        self.classifier = nn.Linear(in_features=1024, out_features=num_class, bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x)  # x_norm_clstoken
        output = self.classifier(token)
        return output



class MIL_ViT_classifier_UNI(ViT_classifier_UNI):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ViT_classifier_UNI, self).__init__(num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output


class ViT_classifier_Phikon(nn.Module):

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):

        from transformers import ViTModel

        super(ViT_classifier_Phikon, self).__init__()
        login(token="")
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

        self.classifier = nn.Linear(in_features=768, out_features=num_class, bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward_backbone(self, x):
        return self.backbone(x).last_hidden_state[:, 0, :]

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x).last_hidden_state[:, 0, :]
        output = self.classifier(token)
        return output



class MIL_ViT_classifier_Phikon(ViT_classifier_Phikon):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ViT_classifier_Phikon, self).__init__(num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch).last_hidden_state[:, 0, :] for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)   # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch).last_hidden_state[:, 0, :] for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output



class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):

        from ctranstimm.models.layers.helpers import to_2tuple

        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class ViT_classifier_CTranspath(nn.Module):

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):


        import ctranstimm

        super(ViT_classifier_CTranspath, self).__init__()
        self.backbone = ctranstimm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
        self.backbone.load_state_dict(torch.load('./downstream/ctranspath.pth', map_location='cpu')['model'], strict=False)
        self.backbone.head = nn.Identity()

        self.classifier = nn.Linear(in_features=768, out_features=num_class, bias=True)
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x)  # x_norm_clstoken
        output = self.classifier(token)
        return output



class MIL_ViT_classifier_CTranspath(ViT_classifier_CTranspath):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ViT_classifier_CTranspath, self).__init__(num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output



class ViT_classifier_HIPT(nn.Module):   # tile级, slide级另外方法

    def __init__(self, num_class, pretrained_path=None, ViT_freeze=True):


        from downstream.HIPT_4K.hipt_4k import HIPT_4K

        super(ViT_classifier_HIPT, self).__init__()
        self.backbone = HIPT_4K().model256.to('cpu')

        self.classifier = nn.Linear(in_features=384, out_features=num_class, bias=True) # 256_384 / 4k_192
        self.classifier.weight.data.fill_(0.0)
        self.classifier.bias.data.fill_(0.0)

        if ViT_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward_backbone(self, x):
        return self.backbone(x)

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        token = self.backbone(x)  # x_norm_clstoken
        output = self.classifier(token)
        return output



class MIL_ViT_classifier_HIPT(ViT_classifier_HIPT):

    def __init__(self, num_class, patch_size, patch_resize, stride, pretrained_path=None, ViT_freeze=True):
        super(MIL_ViT_classifier_HIPT, self).__init__(num_class=num_class, ViT_freeze=ViT_freeze)
        self.patch_size = patch_size
        self.patch_resize = patch_resize
        self.stride = stride

    def forward_backbone(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        return pooling_tokens

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward(self, x):
        patches = extract_patches(x, patch_size=self.patch_size, patch_resize=self.patch_resize, stride=self.stride)  # (patch_num, batch_size, c, patch_size, patch_size)

        tokens = [self.backbone(patch) for patch in patches]  # x_norm_clstoken
        cat_tokens = torch.stack(tokens, dim=0)
        pooling_tokens, _ = torch.max(cat_tokens, dim=0)

        output = self.classifier(pooling_tokens)
        return output
