import os
import csv
from pydub import AudioSegment  # (not used directly, but kept if you need it later)
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.inference import EncoderASR
import sentencepiece as spm

# ---------------- Model setup ----------------
experiment_dir = "/home/jacobala@alabsad.fau.de/New_speechbrain/speechbrain/recipes/CommonVoice/ASR/CTC/NewDataASCMixresults/wavlm_ctc_en/123"
hparams_file  = "/home/jacobala@alabsad.fau.de/New_speechbrain/speechbrain/recipes/CommonVoice/ASR/CTC/NewDataASCMixresults/wavlm_ctc_en/123/hyperparams.yaml"

tokenizer_path = "/home/jacobala@alabsad.fau.de/New_speechbrain/speechbrain/recipes/CommonVoice/ASR/CTC/NewDataASCMixresults/wavlm_ctc_en/123/save/28_char.model"
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(tokenizer_path)

asr_model = EncoderASR.from_hparams(
    source=experiment_dir,
    hparams_file=hparams_file,
    savedir="tmp_dir",
)

print("Loaded ASR model:", asr_model)

# ---------------- Paths ----------------
#audio_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/SLUE_TEST"
#output_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/TestSet_transcript"
audio_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1NewmixsampleResultsSNR15/SNR_15"
output_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/sampleevaluation15dbrate30"
# Outputs
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "1mixedtranscript.csv")

# ---------------- CSV setup ----------------
# We'll (re)write the CSV header once. If you prefer appending, switch mode="a" and
# write the header only when the file doesn't exist.
fieldnames = ["audio_filename", "transcript"]
with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# ---------------- Transcription loop ----------------
processed = 0
failed = 0

for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if not file.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(root, file)
        transcript_filename = os.path.join(
            output_dir, f"speechbrain_retrained_{file[:-4]}.txt"
        )

        try:
            print(f"Transcribing full audio: {audio_path}...")
            result = asr_model.transcribe_file(audio_path).strip()

            # Write per-file TXT
            with open(transcript_filename, "w", encoding="utf-8") as f:
                f.write(result)

            # Append to CSV
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    "audio_filename": file,
                    "transcript": result,
                })

            print(f"Transcript saved to {transcript_filename}")
            processed += 1

        except Exception as e:
            err_msg = str(e)
            print(f"Failed to process {audio_path} due to error: {err_msg}")

            # Log failure to CSV as well (empty transcript, with error message)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    "audio_filename": file,
                    "transcript": ""
                    
                })
            failed += 1

print(f"Done. Processed: {processed}, Failed: {failed}")
print(f"CSV written to: {csv_path}")
