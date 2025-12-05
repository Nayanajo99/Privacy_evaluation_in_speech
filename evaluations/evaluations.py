import os
import pandas as pd
import jiwer
from g2p_en import G2p

# Required only once
g2p = G2p()

# === Settings ===
asr_csv_path = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/11newattackercommonvoice15dbrate30/commonvoice15brate30.csv"
original_transcript_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull"
output_csv_path = "dataretrainedsample15dbrate30.csv"

# === Load ASR CSV ===
asr_df = pd.read_csv(asr_csv_path)

# === Normalize text ===
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# === Extract match_id ===
def extract_id(filename):
    parts = filename.split("_with_")[0] if "_with_" in filename else filename
    return parts.split(".")[0]

asr_df["match_id"] = asr_df["audio_filename"].apply(extract_id)

# === Function to load original .txt transcript ===
def load_original_transcript(match_id):
    txt_file = os.path.join(original_transcript_dir, f"{match_id}.txt")
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            return f.read().strip()
    return None

# === Compare and compute all metrics ===
results = []
for _, row in asr_df.iterrows():
    match_id = row["match_id"]
    asr_text = str(row["transcript"])
    original_text = load_original_transcript(match_id)

    if original_text:
        ref = transform(original_text)
        hyp = transform(asr_text)

        wer = min(jiwer.wer(ref, hyp) * 100, 100)
        cer = min(jiwer.cer(ref, hyp) * 100, 100)
        mer = min(jiwer.mer(ref, hyp) * 100, 100)

        ref_phonemes = ' '.join(g2p(ref))
        hyp_phonemes = ' '.join(g2p(hyp))
        per = min(jiwer.wer(ref_phonemes, hyp_phonemes) * 100, 100)



       

        results.append({
            "match_id": match_id,
            "audio_filename": row["audio_filename"],
            "asr_transcript": asr_text,
            "wer": wer,
            "cer": cer,
            "mer": mer,
            "per": per
        })
    else:
        print(f"Missing original transcript for: {match_id}")

# === Save results ===

results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)
print(f"Final metrics saved to: {output_csv_path}")
# === Compute and print averages ===
avg_wer = results_df["wer"].mean()
avg_cer = results_df["cer"].mean()
avg_mer = results_df["mer"].mean()
avg_per = results_df["per"].mean()

print("\n Average Error Rates Across All Clips:")
print(f"Word Error Rate (WER):     {avg_wer:.4f}")
print(f" Character Error Rate (CER): {avg_cer:.4f}")
print(f" Match Error Rate (MER):     {avg_mer:.4f}")
print(f" Phoneme Error Rate (PER):   {avg_per:.4f}")


