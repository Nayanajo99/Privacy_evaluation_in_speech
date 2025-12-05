import torch
import torchaudio
from hear21passt.base import get_basic_model, get_model_passt
from feature_extraction import AugmentMelSTFT
from passt import PaSST
import os
import csv
import pandas as pd


# Load PaSST model
model = PaSST(AugmentMelSTFT(), nclasses=15)


#checkpoint = torch.load("/tmp/dcase-tut2016-model.pt", map_location=torch.device("cuda"))
checkpoint = torch.load("/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/anjana_masterthesisnew/asc/best-model1.pt", map_location=torch.device("cuda"),weights_only=False)
# Extract the model weights
print('model key',checkpoint['model'].keys())
state_dict = checkpoint['model'] 


# Load the state_dict into model.net
model.load_state_dict(state_dict)
model = model.cuda()
print("model successfully loaded")


original_scene_path = 'Mapping1'
original_scene_dict = {}


# Class labels
class_labels = [
   "beach", "bus", "cafe/restaurant", "car", "city_center", "forest_path", "grocery_store", "home",
   "library", "metro_station", "office", "park", "residential_area", "train", "tram"
]




with open(original_scene_path, 'r') as file:
   for line in file:
       parts = line.strip().split('\t')
       if len(parts) == 2:
           audio_file = parts[0]
           scene_name = parts[1]
           original_scene_dict[audio_file] = scene_name


# Directory with mixed audio files
folder_path = '/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1newoutput_combined_multi_snr/SNR_20'
print("successfully file opened")
# Process files
results = []
print ("processed")


for filename in os.listdir(folder_path):
  
   if filename.endswith(".wav"):
      
       file_path = os.path.join(folder_path, filename)
       waveform, sample_rate = torchaudio.load(file_path)
       print("processing file:")


       if sample_rate != 32000:
           waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=32000).cuda()
       if waveform.size(0) > 1:
           waveform = waveform.mean(dim=0, keepdim=True)
          


       try:
           with torch.no_grad():
               waveform = waveform.to('cuda')
               logits, _, _ = model(waveform)
             
           max_logit_value, max_logit_index = torch.max(logits, dim=1)
           predicted_class_index = max_logit_index.item()
           predicted_class_name = class_labels[predicted_class_index]
           print("predicted:")


           # === Extract scene_id and snr from filename ===
           try:
               scene_id = filename.split('_with_')[1].split('_')[0]  # e.g., 178
               snr_str = filename.split('snr')[-1].replace('.wav', '')  # e.g., -5
               snr = int(snr_str)
           except Exception as e:
               print("failed")
               continue


           original_scene_key = f"audio/{scene_id}.wav"
           original_scene_name = original_scene_dict.get(original_scene_key, "UNKNOWN")
           print("true scene")


           results.append({
               "filename": filename,
               "scene_id": scene_id,
               "snr": snr,
               "max_logit_value": max_logit_value.item(),
               "predicted_class_index": predicted_class_index,
               "predicted_class_name": predicted_class_name,
               "original_class": original_scene_name
           })


       except Exception as e:
           print(f"Failed to process {filename}: {e}")


# Save full results
df = pd.DataFrame(results)
df.to_csv('/home/jacobala@alabsad.fau.de/AOT/1baseevaluationsnr20.csv', index=False)
print("\n Saved full ASC predictions to '1baseevaluationsnr20.csv'")




