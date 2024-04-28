import numpy as np
import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def trunc_or_pad(waveform, target_len):
    sig_len = len(waveform)
    diff_len = abs(sig_len - target_len)

    if (sig_len > target_len):
      # Truncate the signal to the given length
      start_idx = np.random.randint(0, diff_len)
      waveform = waveform[start_idx:start_idx + target_len]

    elif (sig_len < target_len):
      # Length of padding to add at the beginning and end of the signal
      pad1 = np.random.randint(0, diff_len)
      pad2 = diff_len - pad1
      waveform = nn.functional.pad(waveform, pad=(pad1, pad2), mode='constant', value=0)

    return waveform

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

def create_frames(waveform, duration=5, sr=32000):
    frame_size = int(duration * sr)
    waveform = nn.functional.pad(waveform, pad=(0, frame_size - len(waveform)%frame_size)) # pad the end
    waveform = waveform.squeeze()
    frames = waveform.view(-1, frame_size)
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
            top_db = 80
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
        


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        label = torch.tensor(item['target'])
        #label = nn.functional.one_hot(label, num_classes=self.n_classes)

        file = item['filepath']
        waveform, sr = torchaudio.load(file)
        waveform = waveform.squeeze()
        assert len(waveform.shape) == 1, 'Signal with multiple channels detected'
        waveform = trunc_or_pad(waveform, self.audio_len)
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=self.n_fft, win_length=self.window,  hop_length=self.hop_length, 
                                         n_mels=self.n_mels, f_min=self.fmin, f_max=self.fmax)(waveform)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)(spec)
        spec = spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        # Standardize
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
            top_db = 80
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

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            label = torch.tensor(self.targets[idx])

        file = self.files[idx]
        waveform, sr = torchaudio.load(file)
        waveform = waveform.squeeze()
        assert len(waveform.shape) == 1, 'Signal with multiple channels detected'
        frames = create_frames(waveform)
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=self.n_fft, win_length=self.window,  hop_length=self.hop_length, 
                                         n_mels=self.n_mels, f_min=self.fmin, f_max=self.fmax)(frames)
        spec = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)(spec)
        # Standardize
        spec = (spec - spec.mean()) / spec.std()

        # expand to 3 channels for imagenet trained models
        spec = spec.unsqueeze(1).expand(-1,3,-1,-1)

        if self.targets is not None:
            return spec, label
        else:
            return spec, file
    

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

   
