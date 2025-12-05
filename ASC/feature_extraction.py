import torch
from torch import nn
from torch.nn.functional import conv1d, conv2d
import torchaudio
from glob import glob
from functools import lru_cache

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization

class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, n_mfcc=None, sr=32000, win_length=800, hopsize=320, n_fft=1024, 
            htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1, fmax_aug_range=1000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.htk = htk
        self.fmin = fmin

        fmax = None if fmax == "None" else fmax
        if fmax is None:
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)

    def forward(self, x):

        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)

        powspec = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=True)

        powspec = powspec.abs() ** 2
        

        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        # don't augment eval data
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=powspec.device)

        with torch.amp.autocast(enabled=False, device_type="cuda"):
            melspec = torch.matmul(mel_basis, powspec)

        melspec = (melspec + 0.00001).log()
        melspec = (melspec + 4.5) / 5. # fast normalization


        return melspec.unsqueeze(1), None

    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length,
                                               self.hopsize
                                               )

class Mel(nn.Module):
    def __init__(self):
        super(Mel, self).__init__()
        self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                             timem=20, htk=False, fmin=0.0, fmax=16000, norm=1, fmin_aug_range=500,
                             fmax_aug_range=1)

    def forward(self, x):
        specs = self.mel(x)
        #breakpoint()
        specs = specs.unsqueeze(1)
        return specs

class MelTest(nn.Module):
    def __init__(self):
        super(MelTest, self).__init__()

        self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0,
                             timem=0, htk=False, fmin=0.0, fmax=16000, norm=1, fmin_aug_range=1,
                             fmax_aug_range=1, is_training=False)

    def forward(self, x):
        specs = self.mel(x)
        specs = specs.unsqueeze(1)
        return specs

