import pandas as pd
import ast
import re
from collections import defaultdict

# === Load CSV files ===
original_df = pd.read_csv("/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull/all_transcriptsnw.csv")  # Contains: audio_filename, raw_text, raw_ner
asr_df = pd.read_csv("/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1newattackercommonvoice10dbrate20/commonvoice10brate30.csv")  # Contains: audio_filename, transcript


def extract_id(filename):
    match = re.match(r"(clip_\d+_speaker_\d+)", str(filename))
    return match.group(1) if match else filename

original_df["match_id"] = original_df["audio_filename"].apply(extract_id)
asr_df["match_id"] = asr_df["audio_filename"].apply(extract_id)


df = pd.merge(asr_df, original_df, on="match_id", suffixes=("_asr", "_orig"))

# Normalize function
def normalize(text):
    return re.sub(r"[^\w\s]", "", str(text)).lower().strip()


entity_leakage_records = []
total_by_type = defaultdict(int)
leaked_by_type = defaultdict(int)
clip_leakage = defaultdict(bool)  

# === Leakage evaluation ===
for idx, row in df.iterrows():
    raw_text = row.get("raw_text", "")
    asr_transcript = normalize(row.get("transcript", ""))
    match_id = row.get("match_id", "")
    audio_filename = row.get("audio_filename_asr", row.get("audio_filename", ""))
    
  
    try:
        ner_spans = ast.literal_eval(row.get("raw_ner", "[]").replace('""', '"'))
    except Exception as e:
        print(f"[Row {idx}] Error parsing NER: {e}")
        continue

    asr_words = asr_transcript.split()
    seen_entity_norms_in_clip = set()

    for span in ner_spans:
        if isinstance(span, list) and len(span) == 3:
            label, start, length = span
            if isinstance(label, str) and isinstance(start, int) and isinstance(length, int):
                entity_surface = raw_text[start:start + length]
                entity_norm = normalize(entity_surface)

                if not entity_norm:
                    continue

                entity_key = (label, entity_norm)
                if entity_key in seen_entity_norms_in_clip:
                    continue 

                seen_entity_norms_in_clip.add(entity_key)
                total_by_type[label] += 1

                # Check if entity appears in ASR transcript
                entity_tokens = entity_norm.split()
                n = len(entity_tokens)
                leaked = False

                for i in range(len(asr_words) - n + 1):
                    ngram = ' '.join(asr_words[i:i + n])
                    if ngram == entity_norm:
                        leaked = True
                        break

                if leaked:
                    leaked_by_type[label] += 1
                    clip_leakage[match_id] = True

                entity_leakage_records.append({
                    "match_id": match_id,
                    "audio_filename": audio_filename,
                    "entity_type": label,
                    "entity_surface": entity_surface,
                    "leaked": leaked,
                    "asr_transcript": row.get("transcript", "")
                })

# === Save CSVs ===
all_log = pd.DataFrame(entity_leakage_records)
all_log.to_csv("attackerentityleakagereport40.csv", index=False)

leaked_only = all_log[all_log["leaked"] == True]
leaked_only.to_csv("attackerentitynw40.csv", index=False)

# === Clip-level leakage rate ===
total_entities = len(all_log)
leaked_entities = int(all_log["leaked"].sum())  

overall_leak_percent = (leaked_entities / total_entities * 100) if total_entities else 0

print("\n===  Entity-Based Leakage (Overall) ===")
print(f" Total entities (GT) : {total_entities}")
print(f" Leaked entities     : {leaked_entities}")
print(f"Leakage Percent     : {overall_leak_percent:.2f}%")

#leak percent
per_clip = (
    all_log.groupby(["match_id", "audio_filename"])["leaked"]
           .agg(total_entities_in_clip="count", leaked_entities_in_clip="sum")
           .reset_index()
)
per_clip["leak_percent_in_clip"] = (
    per_clip["leaked_entities_in_clip"] / per_clip["total_entities_in_clip"] * 100
)


per_clip.to_csv("attackerclip_leakage_entity_based40.csv", index=False)

print("\n===  Entity-Based Leakage (Per Clip) ===")
print(per_clip[["match_id", "leak_percent_in_clip"]].head(10).to_string(index=False))

TP = sum(1 for r in entity_leakage_records if r["leaked"] == True)
FN = sum(1 for r in entity_leakage_records if r["leaked"] == False)

precision = 1.0  # only evaluating against ground-truth entities
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n===  NER Leakage Evaluation (Recall-focused) ===")
print(f" True Positives (leaked)   : {TP}")
print(f"False Negatives (masked)  : {FN}")
print(f"Precision (assumed 1.0)   : {precision:.2f}")
print(f"Recall (leakage ratio)    : {recall:.4f}")
print(f"F1-Score                  : {f1:.4f}")
