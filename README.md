# Privacy_evaluation_in_speech

The framework evaluates privacy–utility trade-offs for speech mixed with acoustic scenes (ASC) under sampling-based temporal obfuscation. Privacy leakage is measured using classical ASR metrics, semantic embedding-based metrics, and entity-aware named-entity leakage metrics proposed in the thesis.

---

# System Requirements

- Ubuntu 20.04 / 22.04 LTS  
- Sufficient disk space for datasets (≈ 30 GB)  
- To ensure efficient ASR retraining, embedding computations and audio processing:  
  - 8–16 core CPU  
  - 32 GB RAM  
  - NVIDIA GPU with ≥ 8 GB VRAM  
  - CUDA 11.8 or later  

---

# 2. Installation

Install Miniconda.

### Create conda environment:
```bash
conda create -n test python=3.12
conda activate test

pip install -r requirements.txt
```

# Step 1 — Baseline Preparation
1. Prepare Speech Dataset
   ```
   python Audiotrim.py
   ```
   
This concatenates all utterances per speaker and generates the final 390 clean speech files.

4. Mix SLUE speech with ASC background noise at different SNR levels.
     ```
     python SpeechASCMix.py
     ```
      (later SNR with highest utility chosen)

6. For further evaluation in Speech dataset
   ```
   python Vad_frm_csv.py
    ```


7. Apply Sampling-Based Obfuscation
 ```
   python sampling.py
  ```

# Step 2 — Baseline Evaluation
For baseline Evaluation we first check performance of pretrained SpeechBrain and Whisper models
```bash
python speechbrainasr.py
python whispertranscript.py
```
#It generates the ASR transcript and evaluates the WER in evaluations.py
also checks utility evaluation 
```
pip install transformers

python passt_newupdated.py
python confusion_matrx.py
```
where SNR with best utility value is chosen

  retrains the the baseline model
  Retrain the SpeechBrain ASR Model

WavLM/Wav2Vec model on the mixed dataset- ASC + slue speech data at 15 db SNR
```
git clone https://github.com/speechbrain/speechbrain/
cd speechbrain
pip install -r requirements.txt

python wav2vectrain.py hparams/train_en.yaml --device cuda

#use the retrained model for generating ASR transcript
python Retrained_speechbrain.py
```
# Step 3 - Evaluations
use the same model for further evaluations on the audio directory changes
```
  python NERleakage.py
 ```
for evaluating named entity leakage rate

# Embedding
FastText Embeddings
```
Install Gensim: pip install gensim

Download the FastText English embeddings:
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz

Load in Python:
from gensim.models.fasttext import load_facebook_model
 model = load_facebook_model("cc.en.300.bin")

python WER_embed.py #for embedding based WER
python NE_embedding.py #for embedding based named entity error rate
```
# Step 4 - Attacker model Evaluations
Based on different attacker knowledge levels evaluate each
1. ignorant attacker- uses pretrained models
   ```
   python nermodel.py
   python speechbrainasr.py
   ```
3. semi-informed attacker and informed attacker
   ```
    python Retrained_speechbrain_sample.py
   ```
   Here based on attacker knowledge retraining model varies its parameter and datas






