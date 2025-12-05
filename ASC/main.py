import torch
import torchaudio
from hear21passt.base import get_basic_model,get_model_passt
import sys
from feature_extraction import AugmentMelSTFT
from passt import PaSST

# define a dummy class containing PaSST model for loading
# get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
model = PaSST(AugmentMelSTFT(), nclasses=15)


checkpoint = torch.load("/home/jacobala@alabsad.fau.de/AOT/anjana_masterthesisnew/asc/best-model.pt", map_location=torch.device("cuda"))
# Extract the model weights
print('model key',checkpoint['model'].keys())
state_dict = checkpoint['model']  

# Load the state_dict into model.net
model.load_state_dict(state_dict)
model = model.cuda()

print('model_net',model)
# load file
series, fs = torchaudio.load("/home/jacobala@alabsad.fau.de/AOT/output_combinedfinal4")
series = torchaudio.functional.resample(series, orig_freq=fs, new_freq=32000).cuda()

# evaluate model and turn logits into probability estimates
with torch.no_grad():
     pred, _, _ = model(series)
     pred = torch.nn.functional.softmax(pred, dim=-1)
     max_pred, max_index = torch.max(pred, dim=-1)

     breakpoint()


