import torch
import torch.nn as nn
import numpy as np

from torchvision.models import get_model
import timm

class BasicClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super(BasicClassifier, self).__init__()
        weights = 'DEFAULT' if pretrained else None
        self.backbone = get_model(model_name, weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, n_classes)
            )
        
    def forward(self, x, return_dict=False):
        x = self.backbone(x)
        x = self.pool(x).squeeze(dim=(-1,-2))
        x = self.classifier(x)
        return x
    
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


class GeMClassifier(torch.nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True, *args, **kwargs):
        super().__init__()

        out_indices = (3, 4)
        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            out_indices=out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, n_classes)

    def forward(self, x, return_dict=False):
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        return x
    

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = nn.functional.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output

class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class SEDClassifier(nn.Module):
    def __init__(self, num_classes, model_name, n_mels=128, pretrained=True, *args, **kwargs):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(n_mels)

        self.encoder = timm.create_model(
            model_name, 
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,)
            )
        
        feature_dims = self.encoder.feature_info.channels()
        n_channels = sum(feature_dims)
        
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(n_channels, n_channels, bias=True)
        self.att_block = AttBlockV2(
            n_channels, num_classes, activation="sigmoid")


    def forward(self, input_data, return_dict=False):
        x = input_data.transpose(2,3) # (batch_size, 3, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = x.transpose(2, 3)
        x = self.encoder(x)[0]
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = nn.functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = nn.functional.relu(self.fc1(x))
        x = x.transpose(1, 2)
        x = nn.functional.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)


        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'logit': logit,
            'framewise_logit': framewise_logit,
        }

        if return_dict:
            return output_dict
        else:
            return logit