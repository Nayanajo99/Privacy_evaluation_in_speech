import os
import csv
from speechbrain.pretrained import EncoderDecoderASR

# Load the pretrained SpeechBrain ASR model

asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-commonvoice-en",
    savedir="pretrained_models/asr-wav2vec2-commonvoice-en"
)

input_path = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull"
output_csv_path = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/pretrained_model_evaluationoriginalspeech.csv"

results = []

print(f"\n=== Starting SpeechBrain ASR Transcription for Directory ===")
print(f" Input Directory : {input_path}")
print(f" Output CSV       : {output_csv_path}")

# Process all WAV files in the directory

for file in os.listdir(input_path):
    if not file.endswith(".wav"):
        continue

    audio_path = os.path.join(input_path, file)
    base_name = os.path.splitext(file)[0]

    transcript_file = os.path.join(input_path, f"speechbrain_{base_name}.txt")

    try:
        print(f" → Transcribing: {audio_path}")

        # Run SpeechBrain ASR
        transcript = asr_model.transcribe_file(audio_path).strip()

        # Save transcript as text file
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")

        # Prepare CSV row
        results.append({
            "audio_filename": file,
            "transcript": transcript,
        })

    except Exception as e:
        print(f"  Error transcribing {file}: {e}")
      
        results.append({
            "audio_filename": file,
            "transcript": "",
        })


# Save all transcripts to CSV

with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["audio_filename", "transcript"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✓ All transcripts saved to CSV: {output_csv_path}")
print("✓ Processing completed.\n")
