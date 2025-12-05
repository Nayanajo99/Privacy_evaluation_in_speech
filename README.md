# Privacy_evaluation_in_speech

The framework evaluates privacy–utility trade-offs for speech mixed with acoustic scenes (ASC) under sampling-based temporal obfuscation. Privacy leakage is measured using classical ASR metrics, semantic embedding-based metrics, and entity-aware named-entity leakage metrics proposed in the thesis.

1.System Requirements
Ubuntu 20.04 / 22.04 LTS

Sufficient disk space for datasets (≈ 30 GB)
To ensure efficient ASR retraining, embedding computations and audio processing:

8–16 core CPU

32 GB RAM

NVIDIA GPU with ≥ 8 GB VRAM 

CUDA 11.8 or later

2.Installation
install miniconda 
-Create conda environment
conda create -n test python=3.12
conda activate test

pip install -r requirements.txt

Step 1 — Baseline Preparation
1.Prepare Speech Dataset- run python Audiotrim.py
---This concatenates all utterances per speaker and generates the final 390 clean speech files.
2.Mix SLUE speech with ASC background noise at different SNR levels.
--run SpeechASCMix.py (later SNR with highest utility chosen)
3. For further evaluation in Speech dataset - run Vad_frm_csv.py

4. Apply Sampling-Based Obfuscation
   --run sampling.py

Step 2 — Baseline Evaluation
For baseline Evaluation we first check performance of pretrained SpeechBrain and Whisper models
--run speechbrainasr.py
--run whispertranscript.py
It generates the ASR transcript and evaluates the WER in evaluations.py
#also checks utility evaluation 

pip install transformers

--run passt_newupdated.py
--confusion_matrx.py
#where SNR with best utility value is chosen

#retrains the the baseline model
Retrain the SpeechBrain ASR Model

WavLM/Wav2Vec model on the mixed dataset- ASC + slue speech data at 15 db SNR

git clone https://github.com/speechbrain/speechbrain/
cd speechbrain
pip install -r requirements.txt

python wav2vectrain.py hparams/train_en.yaml --device cuda

#use the retrained model for generating ASR transcript
--run Retrained_speechbrain.py

Step 3 - Evaluations
#use the same model for further evaluations on the audio directory changes
#run NERleakage.py for evaluating named entity leakage rate

Embedding
## FastText Embeddings

Install Gensim: pip install gensim

Download the FastText English embeddings:
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz

Load in Python:
from gensim.models.fasttext import load_facebook_model
 model = load_facebook_model("cc.en.300.bin")

#run WER_embed.py for embedding based WER
#run NE_embedding.py for embedding based named entity error rate

Step 4 - Attacker model Evaluations
# Based on different attacker knowledge levels evaluate each
1. ignorant attacker- uses pretrained models
   - run nermodel.py
   - run speechbrainasr.py
2. semi-informed attacker and informed attacker
   - run Retrained_speechbrain_sample.py
   #Here based on attacker knowledge retraining model varies its parameter and datas






