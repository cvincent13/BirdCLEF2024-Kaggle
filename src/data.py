import numpy as np
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Truncate or pad the audio sample to the desired duration
def trunc_or_pad(waveform, audio_len, start_idx='random'):
    sig_len = waveform.shape[-1]
    diff_len = abs(sig_len - audio_len)

    if (sig_len > audio_len):
        # Truncate the signal to the given length
        if start_idx == 'random':
            start_idx = np.random.randint(0, diff_len)
        else:
            start_idx = 0
        waveform = waveform[:, start_idx:start_idx + audio_len]
    
    elif (sig_len < audio_len):
        # Length of padding to add at the beginning and end of the signal
        pad1 = np.random.randint(0, diff_len)
        pad2 = diff_len - pad1
        if isinstance(waveform, torch.Tensor):
            waveform = nn.functional.pad(waveform, pad=(pad1, pad2), mode='constant', value=0)
        else:
            waveform = np.pad(waveform, ((0, 0), (pad1, pad2)), mode='constant', constant_values=0)
    
    return waveform

def create_frames(waveform, duration=5, sr=32000):
    frame_size = int(duration * sr)
    surplus = waveform.size(-1)%frame_size
    if waveform.size(-1) <= surplus:
        waveform = nn.functional.pad(waveform, pad=(0, frame_size - waveform.size(-1)%frame_size), mode='constant', value=0)
    elif surplus > 0:
        waveform = waveform[:, :-surplus]
    frames = waveform.view(-1, 1, frame_size)
    return frames


def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5):
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_, last_state


def find_peak_max(x, filter='savgol'):
    if filter == 'savgol':
        smooth_x = signal.savgol_filter(x, window_length=100, polyorder=2)
    elif filter == 'gaussian':
        smooth_x = gaussian_filter1d(x, sigma=25)
    else:
        smooth_x = x
    return smooth_x.argmax()

def window_around_peak(len_x, peak, window_size):
    half_window = window_size // 2
    start_index = max(0, peak - half_window)
    end_index = min(len_x, peak + half_window)

    # Adjust the window if it's too close to the borders
    if end_index - start_index < window_size:
        if start_index == 0:
            end_index = min(len_x, start_index + window_size)
        elif end_index == len_x:
            start_index = max(0, end_index - window_size)
    return start_index, end_index

class AudioDataset(Dataset):
    def __init__(
            self, 
            df, 
            Config,
            waveform_transforms=None,
            spec_transforms=None,
            ):
        super(AudioDataset, self).__init__()
        self.df = df
        self.n_classes = Config.n_classes
        self.start_idx = Config.start_idx
        self.duration = Config.duration
        self.sample_rate = Config.sample_rate
        self.audio_len = self.duration*self.sample_rate
        self.target_length = Config.target_length
        self.n_mels = Config.n_mels
        self.n_fft = Config.n_fft
        self.window = Config.window
        self.hop_length = self.audio_len // (self.target_length-1) if Config.hop_length is None else Config.hop_length
        self.fmin = Config.fmin
        self.fmax = Config.fmax
        self.top_db = Config.top_db
        self.standardize = Config.standardize
        self.mean = Config.dataset_mean
        self.std = Config.dataset_std
        self.loss = Config.loss
        self.secondary_labels_weight = Config.secondary_labels_weight
        self.n_channels = Config.n_channels
        self.use_1_peak = Config.use_1_peak
        self.peak_filter = Config.peak_filter
        self.use_peaks = Config.use_peaks

        self.to_mel_spectrogramn = torchaudio.transforms.MelSpectrogram(self.sample_rate, n_fft=self.n_fft, win_length=self.window,  
                                                 hop_length=self.hop_length, n_mels=self.n_mels, 
                                                 f_min=self.fmin, f_max=self.fmax)

        self.mel_to_db = nn.Sequential(torchaudio.transforms.AmplitudeToDB(top_db=self.top_db))

        if self.mean is not None and self.std is not None:
            self.mel_to_db.append(v2.Normalize(mean=self.mean, std=self.std))

        self.waveform_transforms = waveform_transforms
        self.spec_transforms  = spec_transforms
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        label = torch.tensor(item['target'])
        if self.loss == 'bce':
            label = nn.functional.one_hot(label, num_classes=self.n_classes).float()
            for l in item['secondary_targets']:
                if l is not None:
                    label += nn.functional.one_hot(torch.tensor(l), num_classes=self.n_classes)*self.secondary_labels_weight

        file = item['filepath']
        waveform, sr = torchaudio.load(file)
        waveform = trunc_or_pad(waveform[0], self.audio_len, self.start_idx)

        if self.waveform_transforms is not None:
            waveform = self.waveform_transforms(waveform.squeeze().numpy(), sr)[None,:] 
            waveform = torch.Tensor(waveform)

        spec = self.to_mel_spectrogramn(waveform)

        if self.use_1_peak:
            per_frame_energy = spec.sum(dim=1).squeeze().numpy()
            peak = find_peak_max(per_frame_energy, filter=self.peak_filter)
            start_index, end_index = window_around_peak(len(per_frame_energy), peak, self.target_length)
            spec = spec[:,:,start_index:end_index]
        
        elif self.use_peaks:
            per_frame_energy = spec.sum(dim=1).squeeze().numpy()
            peak1 = find_peak_max(per_frame_energy, filter=self.peak_filter)
            start_index, end_index = window_around_peak(len(per_frame_energy), peak, self.target_length)
            spec1 = spec[:,:,start_index:end_index]


        spec = self.mel_to_db(spec)

        if self.spec_transforms is not None:
            spec = self.spec_transforms(spec)

        # Standardize
        if self.standardize:
            spec = (spec - spec.mean()) / spec.std()

        # expand to 3 channels for imagenet trained models
        if self.n_channels > 1:
            spec = spec.expand(self.n_channels,-1,-1)

        return spec, label
    

class AudioDatasetInference(Dataset):
    def __init__(
            self, 
            files,
            targets = None, 
            n_classes = 182,
            duration = 5,
            sample_rate = 32000,
            target_length = 384,
            n_mels = 128,
            n_fft = 2028,
            window = 2028,
            hop_length = None,
            fmin = 20,
            fmax = 16000,
            top_db = 80,
            standardize=True,
            mean=None,
            std=None
            ):
        super(AudioDatasetInference, self).__init__()
        self.files = files
        self.targets = targets
        self.n_classes = n_classes
        self.duration = duration
        self.sample_rate = sample_rate
        self.audio_len = duration*sample_rate
        self.target_length = target_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.window = window
        self.hop_length = self.audio_len // (target_length-1) if not hop_length else hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db
        self.standardize = standardize

        self.to_mel_spectrogramn = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(self.sample_rate, n_fft=self.n_fft, win_length=self.window,  
                                                 hop_length=self.hop_length, n_mels=self.n_mels, 
                                                 f_min=self.fmin, f_max=self.fmax),
            torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)
        )
        if mean is not None and std is not None:
            self.to_mel_spectrogramn.append(v2.Normalize(mean=mean, std=std))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        file = self.files[idx]
        waveform, sr = torchaudio.load(file)
        frames = create_frames(waveform)[:100]
        spec = self.to_mel_spectrogramn(frames)
        # Standardize
        if self.standardize:
            spec = (spec - spec.mean()) / spec.std()

        # expand to 3 channels for imagenet trained models
        spec = spec.expand(-1,3,-1,-1)

        if self.targets is not None:
            label = torch.tensor(self.targets[idx])
            return spec, label
        else:
            return spec, file
    


class FrequencyMaskingAug(torchaudio.transforms.FrequencyMasking):
    def __init__(self, prob, max_mask_pct, n_mels, n_masks, mask_mode='mean'):
        self.prob = prob
        self.freq_mask_param = max_mask_pct * n_mels
        self.n_masks = n_masks
        self.mask_mode = mask_mode
        super(FrequencyMaskingAug, self).__init__(self.freq_mask_param)
    def forward(self, specgram):
        if self.mask_mode == 'mean':
            mask_value = specgram.mean()
        else:
            mask_value = 0
        
        for _ in range(self.n_masks):
            if np.random.random() < self.prob:
                specgram = super().forward(specgram, mask_value)

        return specgram
    
class TimeMaskingAug(torchaudio.transforms.TimeMasking):
    def __init__(self, prob, max_mask_pct, n_steps, n_masks, mask_mode='mean'):
        self.prob = prob
        self.time_mask_param = max_mask_pct * n_steps
        self.n_masks = n_masks
        self.mask_mode = mask_mode
        super(TimeMaskingAug, self).__init__(self.time_mask_param)
    def forward(self, specgram):
        if self.mask_mode == 'mean':
            mask_value = specgram.mean()
        else:
            mask_value = 0
        
        for _ in range(self.n_masks):
            if np.random.random() < self.prob:
                specgram = super().forward(specgram, mask_value)

        return specgram
    

from typing import Any, Callable, Dict, List, Tuple
import math
import numbers
import warnings
import PIL.Image
from torchvision.transforms.v2._utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F
class _BaseMixUpCutMix(v2.Transform):
    def __init__(self, *, alpha: float = 1.0, num_classes: int, labels_getter="default", one_hot_labels: bool = False) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

        self.num_classes = num_classes

        self._labels_getter = _parse_labels_getter(labels_getter)
        self.one_hot_labels = one_hot_labels

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
            raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        elif labels.ndim != 1 and not self.one_hot_labels:
            raise ValueError(
                f"labels tensor should be of shape (batch_size,) " f"but got shape {labels.shape} instead."
            )
        elif labels.ndim != 2 and labels.size(-1) != self.num_classes and self.one_hot_labels:
            raise ValueError(
                f"labels tensor should be of shape (batch_size, {self.num_classes}) " f"but got shape {labels.shape} instead."
            )

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        if not self.one_hot_labels:
            label = one_hot(label, num_classes=self.num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


class MixUp(_BaseMixUpCutMix):
    """Apply MixUp to the provided batch of images and labels.

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]

        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt


class CutMix(_BaseMixUpCutMix):
    """Apply CutMix to the provided batch of images and labels.

    Paper: `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/abs/1905.04899>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``CutMix()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_size(flat_inputs)

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return dict(box=box, lam_adjusted=lam_adjusted)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=params["lam_adjusted"])
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            x1, y1, x2, y2 = params["box"]
            rolled = inpt.roll(1, 0)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt