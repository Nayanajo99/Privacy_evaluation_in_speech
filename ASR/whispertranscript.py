import os
import csv
import whisper
import soundfile as sf

# === Load model ===
model = whisper.load_model("base")

# === Config ===
input_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull"
transcript_txt_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/whisperoutputoriginalspeech"
output_csv = "originalspeech_whisper.csv"
language = "en"

os.makedirs(transcript_txt_dir, exist_ok=True)
results = []


for file in sorted(os.listdir(input_dir)):
    if file.endswith(".wav"):
        audio_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        txt_path = os.path.join(transcript_txt_dir, f"{base_name}.txt")

        try:
            # Log duration
            y, sr = sf.read(audio_path)
            duration = len(y) / sr
            print(f" Transcribing: {file} | Duration: {duration:.2f} sec")

            # Transcribe with Whisper
            result = model.transcribe(audio_path, language=language)
            transcript = result.get("text", "").strip()

          
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            results.append({
                "audio_filename": file,
                "transcript": transcript
            })

        except Exception as e:
            print(f" Error in {file}: {e}")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("TRANSCRIPTION FAILED: " + str(e))

            results.append({
                "audio_filename": file,
                "transcript": f"FAILED: {str(e)}"
            })

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["audio_filename", "transcript"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone! All transcripts saved in {transcript_txt_dir}")
print(f" Summary CSV saved as: {output_csv}")
