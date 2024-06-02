import torch
import torch.nn as nn
import numpy as np

from torchvision.models import get_model
import timm
from src.efficientat.dymn.model import get_model as get_dymn
from src.efficientat.mn.model import get_model as get_mn
from src.efficientat.utils import NAME_TO_WIDTH
from src.efficientat.mn.attention_pooling import MultiHeadAttentionPooling


class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = self.pool(x.clamp(min=self.eps).pow(self.p)).pow(1.0 / self.p)
        x = x.view(bs, ch)
        return x

class GeMMultiHeadAttentionPooling(nn.Module):
    """Multi-Head Attention as used in PSLA paper (https://arxiv.org/pdf/2102.01243.pdf)
    """
    def __init__(self, in_dim, out_dim, att_activation: str = 'sigmoid',
                 clf_activation: str = 'ident', num_heads: int = 4, epsilon: float = 1e-7):
        super(GeMMultiHeadAttentionPooling, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.epsilon = epsilon

        self.att_activation = att_activation
        self.clf_activation = clf_activation

        p = 3
        self.p1 = torch.nn.Parameter(torch.ones(1) * p)
        self.p2 = torch.nn.Parameter(torch.ones(1) * p)

        # out size: out dim x 2 (att and clf paths) x num_heads
        self.subspace_proj = nn.Linear(self.in_dim, self.out_dim * 2 * self.num_heads)
        self.head_weight = nn.Parameter(torch.tensor([1.0 / self.num_heads] * self.num_heads).view(1, -1, 1))

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return nn.functional.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return nn.functional.softmax(x, dim=1)
        elif activation == 'ident':
            return x

    def forward(self, x):
        """x: Tensor of size (batch_size, channels, frequency bands, sequence length)
        """
        x = torch.mean(x.clamp(min=self.epsilon).pow(self.p1), dim=2).pow(1.0 / self.p1)  # results in tensor of size (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)  # results in tensor of size (batch_size, sequence_length, channels)
        b, n, c = x.shape

        x = self.subspace_proj(x).reshape(b, n, 2, self.num_heads, self.out_dim).permute(2, 0, 3, 1, 4)
        att, val = x[0], x[1]
        val = self.activate(val, self.clf_activation)
        att = self.activate(att, self.att_activation)
        att = torch.clamp(att, self.epsilon, 1. - self.epsilon)
        att = att / torch.sum(att, dim=2, keepdim=True)

        out = torch.sum(att * val, dim=2) * self.head_weight
        out = torch.mean(out.clamp(min=self.epsilon).pow(self.p2), dim=1).pow(1.0 / self.p2)
        return out


class AudioClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super(AudioClassifier, self).__init__()
        width_mult = NAME_TO_WIDTH(model_name)
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=None)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=None)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0., inplace=True),
            nn.Linear(960, n_classes)
            )
        
    def forward(self, x, return_dict=False):
        x, _ = self.backbone(x)
        x = self.pool(x).squeeze(dim=(-1,-2))
        x = self.classifier(x)
        return x
    

class AudioGeMClassifier(torch.nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = (10,12,16)  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(80*width_mult),int(112*width_mult),int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, n_classes)

    def forward(self, x, return_dict=False):
        _, ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        return x
    

class AudioAttClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = None  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.mid_features = np.sum(feature_dims)
        self.att_pool = MultiHeadAttentionPooling(in_dim=self.mid_features, out_dim=n_classes, num_heads=4)

    def forward(self, x, return_dict=False):
        x, _ = self.backbone(x)
        x = self.att_pool(x)
        return x
    
class AudioGeMAttClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = None  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.mid_features = np.sum(feature_dims)
        self.att_pool = GeMMultiHeadAttentionPooling(in_dim=self.mid_features, out_dim=n_classes, num_heads=4)

    def forward(self, x, return_dict=False):
        x, _ = self.backbone(x)
        x = self.att_pool(x)
        return x


class AudioAttMSClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = (10,12,16)  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(80*width_mult),int(112*width_mult),int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.att_pool1 = MultiHeadAttentionPooling(in_dim=feature_dims[0], out_dim=n_classes, num_heads=2)
        self.att_pool1_w = nn.Parameter(torch.tensor(1.))
        self.att_pool2 = MultiHeadAttentionPooling(in_dim=feature_dims[1], out_dim=n_classes, num_heads=2)
        self.att_pool2_w = nn.Parameter(torch.tensor(1.))
        self.att_pool3 = MultiHeadAttentionPooling(in_dim=feature_dims[2], out_dim=n_classes, num_heads=2)
        self.att_pool3_w = nn.Parameter(torch.tensor(1.))

    def forward(self, x, return_dict=False):
        _, features = self.backbone(x)
        x1 = self.att_pool1(features[0])
        x2 = self.att_pool2(features[1])
        x3 = self.att_pool3(features[2])
        x = x1 * self.att_pool1_w + x2 * self.att_pool2_w + x3 * self.att_pool3_w
        return x
    

class AudioGeMAttMSClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = (10,12,16)  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(80*width_mult),int(112*width_mult),int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.att_pool1 = GeMMultiHeadAttentionPooling(in_dim=feature_dims[0], out_dim=n_classes, num_heads=2)
        self.att_pool1_w = nn.Parameter(torch.tensor(1.))
        self.att_pool2 = GeMMultiHeadAttentionPooling(in_dim=feature_dims[1], out_dim=n_classes, num_heads=2)
        self.att_pool2_w = nn.Parameter(torch.tensor(1.))
        self.att_pool3 = GeMMultiHeadAttentionPooling(in_dim=feature_dims[2], out_dim=n_classes, num_heads=2)
        self.att_pool3_w = nn.Parameter(torch.tensor(1.))

    def forward(self, x, return_dict=False):
        _, features = self.backbone(x)
        x1 = self.att_pool1(features[0])
        x2 = self.att_pool2(features[1])
        x3 = self.att_pool3(features[2])
        x = x1 * self.att_pool1_w + x2 * self.att_pool2_w + x3 * self.att_pool3_w
        return x
    

def interpolate(x: torch.Tensor, ratio: int):
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    output = nn.functional.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output
        

class AudioSEDClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = None  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        
        n_channels = sum(feature_dims)
        
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(n_channels),
            nn.ReLU()
        )

        self.att = nn.Conv1d(in_channels=n_channels, out_channels=n_classes, kernel_size=1)
        self.cla = nn.Conv1d(in_channels=n_channels, out_channels=n_classes, kernel_size=1)

    def forward(self, x, return_dict=False):
        # (batch, channels, frequency bands, sequence length)
        x, _ = self.backbone(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=2)

        # Mix temporal axis (frames)
        x1 = nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = self.conv1x1(x)

        # Attention pooling (and keep attention weights)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.cla(x)
        logit = torch.sum(norm_att * cla, dim=2)
        if return_dict:
            segmentwise_logit = cla.transpose(1, 2)
            output_dict = {
                'logit': logit,
                'framewise_logit': segmentwise_logit,
            }
            return output_dict
        else:
            return logit
        

class GeMSEDPooling(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        p = 3
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.epsilon = 1e-6

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(n_channels),
            nn.ReLU()
        )

        self.att = nn.Conv1d(n_channels, n_classes, kernel_size=1)
        self.cla = nn.Conv1d(n_channels, n_classes, kernel_size=1)


    def forward(self, x):
        # Aggregate in frequency axis
        x = torch.mean(x.clamp(min=self.epsilon).pow(self.p), dim=2).pow(1.0 / self.p)

        # Mix temporal axis (frames)
        x1 = nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = self.conv1x1(x)

        # Attention pooling (and keep attention weights)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.cla(x)
        logit = torch.sum(norm_att * cla, dim=2)
        return logit, norm_att, cla
    

class AudioGeMSEDClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = None  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        
        n_channels = sum(feature_dims)
        self.pool = GeMSEDPooling(n_channels, n_classes)

    def forward(self, x, return_dict=False):
        # (batch, channels, frequency bands, sequence length)
        x, _ = self.backbone(x)
        logit, norm_att, cla = self.pool(x)
        if return_dict:
            segmentwise_logit = cla.transpose(1, 2)
            output_dict = {
                'logit': logit,
                'framewise_logit': segmentwise_logit,
            }
            return output_dict
        else:
            return logit
        
class AudioMSGeMSEDClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = (10,12,16)   
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(80*width_mult),int(112*width_mult),int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        
        self.pools = torch.nn.ModuleList([GeMSEDPooling(dim, dim) for dim in feature_dims])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, n_classes)

    def forward(self, x, return_dict=False):
        # (batch, channels, frequency bands, sequence length)
        _, ms = self.backbone(x)
        h = torch.cat([pool(m)[0] for m, pool in zip(ms, self.pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        return x

class MultiHeadGeMSEDPooling(nn.Module):
    def __init__(self, n_channels, out_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        p = 3
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.epsilon = 1e-6

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(n_channels),
            nn.ReLU()
        )

        self.subspace_proj = nn.Linear(n_channels, self.out_dim * 2 * self.num_heads)
        self.head_weight = nn.Parameter(torch.tensor([1.0 / self.num_heads] * self.num_heads).view(1, -1, 1))

    def forward(self, x):
        # Aggregate in frequency axis
        x = torch.mean(x.clamp(min=self.epsilon).pow(self.p), dim=2).pow(1.0 / self.p)

        # Mix temporal axis (frames)
        x1 = nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = self.conv1x1(x)

        # Attention pooling (and keep attention weights)
        x = x.transpose(1, 2)
        b, n, c = x.shape
        x = self.subspace_proj(x).reshape(b, n, 2, self.num_heads, self.out_dim).permute(2, 0, 3, 1, 4)
        att, cla = x[0], x[1]
        norm_att = torch.softmax(torch.tanh(att), dim=-1)
        logit = torch.sum(att * cla, dim=2) * self.head_weight
        logit = torch.sum(logit, dim=1)
        return logit, norm_att, cla


class AudioMultiHeadGeMSEDClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = None  
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(960*width_mult)]      
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        
        n_channels = sum(feature_dims)
        self.pool = MultiHeadGeMSEDPooling(n_channels, n_classes, num_heads=4)


    def forward(self, x, return_dict=False):
        # (batch, channels, frequency bands, sequence length)
        x, _ = self.backbone(x)
        logit, norm_att, cla = self.pool(x)
        if return_dict:
            segmentwise_logit = cla.transpose(1, 2)
            output_dict = {
                'logit': logit,
                'framewise_logit': segmentwise_logit,
            }
            return output_dict
        else:
            return logit
        
    
class AudioMSMultiHeadGeMSEDClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = (10,12,16)   
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(80*width_mult),int(112*width_mult),int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.pools = torch.nn.ModuleList([MultiHeadGeMSEDPooling(dim, n_classes, num_heads=4) for dim in feature_dims])

    def forward(self, x, return_dict=False):
        # (batch, channels, frequency bands, sequence length)
        _, ms = self.backbone(x)
        logits = torch.stack([pool(m)[0] for m, pool in zip(ms, self.pools)], dim=0)
        logit = logits.sum(0)
        return logit
        
class AudioMSMultiHeadGeMSEDClassifier2(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()
        out_indices = (10,12,16)   
        width_mult = NAME_TO_WIDTH(model_name)
        feature_dims = [int(80*width_mult),int(112*width_mult),int(960*width_mult)]
        pretrained_name = model_name if pretrained else None
        if model_name.startswith("dymn"):
            self.backbone = get_dymn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)
        elif model_name.startswith("mn"):
            self.backbone = get_mn(width_mult=width_mult, pretrained_name=pretrained_name, num_classes=n_classes, 
                                    features_only=True, out_indices=out_indices)

        self.pools = torch.nn.ModuleList([MultiHeadGeMSEDPooling(dim, dim, num_heads=1) for dim in feature_dims])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, n_classes)

    def forward(self, x, return_dict=False):
        # (batch, channels, frequency bands, sequence length)
        _, ms = self.backbone(x)
        h = torch.cat([pool(m)[0] for m, pool in zip(ms, self.pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        return x