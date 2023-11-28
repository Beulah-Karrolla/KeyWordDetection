import torch
import torchaudio
import pandas as pd
import speechbrain as sb
from speechbrain.processing.features import STFT, spectral_magnitude, Filterbank, Deltas, InputNormalization, ContextWindow
import torch.nn as nn
from sklearn.utils import resample 
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class SpecAugment(nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()

        self.rate = rate

        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )

        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        policies = { 1: self.policy1, 2: self.policy2, 3: self.policy3 }
        self._forward = policies[policy]

    def forward(self, x):
        return self._forward(x)

    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug(x)
        return x

    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return  self.specaug2(x)
        return x

    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)
    
class RandomCut(nn.Module):
    """Augmentation technique that randomly cuts start or end of audio"""

    def __init__(self, max_cut=10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x):
        """Randomly cuts from start or end of batch"""
        side = torch.randint(0, 1, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            return x[:-cut,:,:]
        elif side == 1:
            return x[cut:,:,:] 
        

class SpeechCommandsDataset(torch.utils.data.Dataset):
    def upsample_minority(self, data):
        df_majority = data[(data['Class']==0)]
        df_minority = data[(data['Class']==1)]
        big_len = len(df_majority)
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=big_len, random_state=42)
        data = pd.concat([df_majority, df_minority_upsampled])
        return data

    def downsample_majority(self, data):
        df_majority = data[(data['Class']==0)]
        df_minority = data[(data['Class']==1)]
        small_len = len(df_minority)
        df_majority_downsampled = resample(df_majority, replace=True, n_samples=small_len, random_state=42)
        data = pd.concat([df_minority, df_majority_downsampled])
        return data

    def __init__(self, data_path, model_type, sample_rate=16000):
        self.data = pd.read_csv(data_path)
        if model_type == 'upsample_minority':
            self.data = self.upsample_minority(self.data)
        elif model_type == 'downsample_majority':
            self.data = self.downsample_majority(self.data)
        #import ipdb;ipdb.set_trace()
        self.sr = sample_rate
        self.filterbank = Filterbank(n_mels=40)
        self.stft = STFT(sample_rate=sample_rate, win_length=25, hop_length=10, n_fft=400)
        self.deltas = Deltas(input_size=40)
        self.input_norm = InputNormalization()

    def __len__(self):
        return self.data.shape[0]
    
    def fix_path(self, path):
        return path.replace('/home/karrolla.1/', '/homes/2/karrolla.1/')

    def __getitem__(self, index):
        curr = self.data.iloc[index]
        wavform, sr = torchaudio.load(self.fix_path(curr['AudioPath']))
        #wavform = self.input_norm(wavform)
        wavform = wavform.type('torch.FloatTensor')
        if sr > self.sr:
            wavform = torchaudio.transforms.Resample(sr, self.sr)(wavform)
        features = self.stft(wavform)
        features = spectral_magnitude(features)
        features = self.filterbank(features)
        return features, curr['Class'], curr['AudioPath']
    
    

def collate_fn(data):
    fbanks = []
    pholders = []
    labels = []
    for d in data:
        fbank, label, pholder = d
        fbanks.append(fbank.squeeze(0) if fbank.size(1) > 2 else fbank)
        labels.append(label)
        pholders.append(pholder)
    
    fbanks = nn.utils.rnn.pad_sequence(fbanks, batch_first=True)  # batch, seq_len, feature
    labels = torch.tensor(labels)
    
    return fbanks, labels, pholders
  
