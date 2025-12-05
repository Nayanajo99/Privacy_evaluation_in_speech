import os
import pandas as pd
import jiwer
from g2p_en import G2p


asr_transcripts_dir = "/home/jacobala@alabsad.fau.de/New_speechbrain/NewmixamapledSNR15rate30"


#   clip_0272_speaker_124715_with_50_snr15.txt
original_transcript_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull"


output_csv_path = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/data15dbin30csv"



g2p = G2p()

# Normalization 
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

# Load ASR transcripts
rows = []
for root, _, files in os.walk(asr_transcripts_dir):
    for fname in files:
        if fname.startswith("speechbrain_retrained_") and fname.endswith(".txt"):
            fpath = os.path.join(root, fname)
            
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()
            except UnicodeDecodeError:
                with open(fpath, "r", encoding="latin-1") as f:
                    transcript = f.read().strip()

            
            # "speechbrain_retrained_<audio_filename>.txt" 
            audio_filename = fname[len("speechbrain_retrained_"):-len(".txt")]

            rows.append({
                "audio_filename": audio_filename,
                "transcript": transcript,
            })

if not rows:
    raise RuntimeError(f"No ASR transcripts found in folder: {asr_transcripts_dir}")

asr_df = pd.DataFrame(rows)

def extract_id(filename: str) -> str:
    
    parts = filename.split("_with_")[0] if "_with_" in filename else filename
    return parts.split(".")[0]

def load_original_transcript(match_id: str):
    
    txt_file = os.path.join(original_transcript_dir, f"{match_id}.txt")
    if os.path.exists(txt_file):
        
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except UnicodeDecodeError:
            with open(txt_file, "r", encoding="latin-1") as f:
                return f.read().strip()
    return None

asr_df["match_id"] = asr_df["audio_filename"].apply(extract_id)

#compute metrics
results = []
missing_gt = 0

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

        # PER via G2P -> WER on phoneme strings
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
            "per": per,
        })
    else:
        missing_gt += 1
        print(f"Missing original transcript for: {match_id}")

if not results:
    #
    pd.DataFrame(columns=["match_id", "audio_filename", "asr_transcript", "wer", "cer", "mer", "per"]).to_csv(
        output_csv_path, index=False
    )
    print(f"No pairs with ground-truth found. Created empty CSV with headers at: {output_csv_path}")
else:
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Final metrics saved to: {output_csv_path}")

    # Averages across available rows
    avg_wer = results_df["wer"].mean()
    avg_cer = results_df["cer"].mean()
    avg_mer = results_df["mer"].mean()
    avg_per = results_df["per"].mean()

    print("\nAverage Error Rates Across All Clips:")
    print(f"  Word Error Rate (WER):      {avg_wer:.4f}")
    print(f"  Character Error Rate (CER): {avg_cer:.4f}")
    print(f"  Match Error Rate (MER):     {avg_mer:.4f}")
    print(f"  Phoneme Error Rate (PER):   {avg_per:.4f}")

if missing_gt:
    print(f"\nNote: {missing_gt} item(s) had no ground-truth transcript and were skipped from scoring.")
