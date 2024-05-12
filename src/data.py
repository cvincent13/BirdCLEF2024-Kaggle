import numpy as np
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

def trunc_or_pad(waveform, audio_len):
    sig_len = waveform.shape[-1]
    diff_len = abs(sig_len - audio_len)

    if (sig_len > audio_len):
        # Truncate the signal to the given length
        start_idx = np.random.randint(0, diff_len)
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


class AudioDataset(Dataset):
    def __init__(
            self, 
            df, 
            n_classes,
            duration = 10,
            sample_rate = 32000,
            target_length = 384,
            n_mels = 128,
            n_fft = 2028,
            window = 2028,
            hop_length = None,
            fmin = 20,
            fmax = 16000,
            top_db = 80,
            waveform_transforms=None,
            spec_transforms=None,
            standardize=True,
            mean=None,
            std=None,
            loss='crossentropy',
            secondary_labels_weight=0.
            ):
        super(AudioDataset, self).__init__()
        self.df = df
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
        self.loss = loss
        self.secondary_labels_weight = secondary_labels_weight

        self.to_mel_spectrogramn = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(self.sample_rate, n_fft=self.n_fft, win_length=self.window,  
                                                 hop_length=self.hop_length, n_mels=self.n_mels, 
                                                 f_min=self.fmin, f_max=self.fmax),
            torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)
        )
        if mean is not None:
            self.to_mel_spectrogramn.append(v2.Normalize(mean=mean, std=std))

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
        waveform = trunc_or_pad(waveform, self.audio_len)

        if self.waveform_transforms is not None:
            waveform = self.waveform_transforms(waveform.numpy(), sr)
            waveform = torch.Tensor(waveform)

        spec = self.to_mel_spectrogramn(waveform)

        if self.spec_transforms is not None:
            spec = self.spec_transforms(spec)

        # Standardize
        if self.standardize:
            spec = (spec - spec.mean()) / spec.std()

        # expand to 3 channels for imagenet trained models
        spec = spec.expand(3,-1,-1)

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
        if mean is not None:
            self.to_mel_spectrogramn.append(v2.Normalize(mean=mean, std=std))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            label = torch.tensor(self.targets[idx])

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


   
def alternative():
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, 
                                              window_type='hanning', num_mel_bins=256, dither=0.0, frame_shift=10)
   
    target_length = 1024
    n_frames = fbank.shape[0]

    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    return None