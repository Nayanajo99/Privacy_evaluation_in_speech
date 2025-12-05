import pandas as pd
import ast
import re
from collections import defaultdict

# === Load CSV files ===
original_df = pd.read_csv(
    "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull/all_transcriptsnw.csv"
)  # columns: audio_filename, raw_text, raw_ner
asr_df = pd.read_csv(
    "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/11NewmixsamapledSNR15rate40/retrainsamplerate40Transcript.csv"
) 

def extract_id(filename):
    match = re.match(r"(clip_\d+_speaker_\d+)", str(filename))
    return match.group(1) if match else filename

original_df["match_id"] = original_df["audio_filename"].apply(extract_id)
asr_df["match_id"] = asr_df["audio_filename"].apply(extract_id)


df = pd.merge(asr_df, original_df, on="match_id", suffixes=("_asr", "_orig"))

# Normalize
def normalize(text):
    return re.sub(r"[^\w\s]", "", str(text)).lower().strip()

#  Helpers for alignment 
_word_re = re.compile(r"\w+", flags=re.UNICODE)

def tokenize_with_spans(text):
    """Return (words, spans) where spans[i]=(start,end) char indices in original text."""
    words, spans = [], []
    for m in _word_re.finditer(text or ""):
        words.append(m.group(0))
        spans.append((m.start(), m.end()))
    return words, spans

def align_words(gt_words, asr_words):
    """
    Word-level Levenshtein alignment.
    Returns list of (i, j, op) with op in {"MATCH","SUB","DEL","INS"}.
    i/j are -1 when the step is a pure insertion/deletion.
    """
    n, m = len(gt_words), len(asr_words)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = "DEL"
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = "INS"

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if gt_words[i-1] == asr_words[j-1] else 1
            del_cost = dp[i-1][j] + 1
            ins_cost = dp[i][j-1] + 1
            sub_cost = dp[i-1][j-1] + cost
            if del_cost <= ins_cost and del_cost <= sub_cost:
                dp[i][j], bt[i][j] = del_cost, "DEL"
            elif ins_cost <= sub_cost:
                dp[i][j], bt[i][j] = ins_cost, "INS"
            else:
                dp[i][j], bt[i][j] = sub_cost, ("MATCH" if cost == 0 else "SUB")

    path = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "DEL":
            path.append((i-1, j-1 if j > 0 else -1, "DEL"))
            i -= 1
        elif op == "INS":
            path.append((i-1 if i > 0 else -1, j-1, "INS"))
            j -= 1
        else:  # MATCH or SUB
            path.append((i-1, j-1, op))
            i -= 1
            j -= 1
    path.reverse()
    return path

def build_gt_to_asr_index_map(path):
    """Map each GT word index to a list of aligned ASR word indices via MATCH/SUB."""
    mp = defaultdict(list)
    for i, j, op in path:
        if i >= 0 and j >= 0 and op in ("MATCH", "SUB"):
            mp[i].append(j)
    for k in list(mp.keys()):
        mp[k] = sorted(set(mp[k]))
    return mp

# Initialize 
entity_leakage_records = []
total_by_type = defaultdict(int)
leaked_by_type = defaultdict(int)
clip_leakage = defaultdict(bool)  

#Leakage evaluation with alignment
for idx, row in df.iterrows():
    raw_text = row.get("raw_text", "")
    asr_transcript = normalize(row.get("transcript", ""))
    match_id = row.get("match_id", "")
    audio_filename = row.get("audio_filename_asr", row.get("audio_filename", ""))

    
    raw_ner_val = row.get("raw_ner", "[]")
    if pd.isna(raw_ner_val):
        raw_ner_val = "[]"
    try:
        ner_spans = ast.literal_eval(str(raw_ner_val).replace('""', '"'))
        if not isinstance(ner_spans, (list, tuple)):
            ner_spans = []
    except Exception as e:
        print(f"[Row {idx}] Error parsing NER: {e}")
        continue

    
    gt_words_raw, gt_spans = tokenize_with_spans(raw_text or "")  #gt= ground_truth
    gt_words = [normalize(w) for w in gt_words_raw]
    asr_words = asr_transcript.split()

    # Alignment path + Ground truth
    align_path = align_words(gt_words, asr_words)
    gt2asr = build_gt_to_asr_index_map(align_path)

    seen_entity_norms_in_clip = set()

    for span in ner_spans:
        if not (isinstance(span, (list, tuple)) and len(span) == 3):
            continue
        label, start, length = span
        if not (isinstance(label, str) and isinstance(start, int) and isinstance(length, int)):
            continue

        entity_surface = raw_text[start:start + length]
        entity_norm = normalize(entity_surface)
        if not entity_norm:
            continue

        
        entity_key = (label, entity_norm)
        if entity_key in seen_entity_norms_in_clip:
            continue
        seen_entity_norms_in_clip.add(entity_key)

        total_by_type[label] += 1

       
        ent_start, ent_end = start, start + length
        ent_gt_word_idxs = [wi for wi, (s, e) in enumerate(gt_spans) if not (e <= ent_start or s >= ent_end)]

        leaked = False
        if ent_gt_word_idxs:
            # Collect ASR indices aligned to those GT words 
            aligned_asr_idxs = []
            for wi in ent_gt_word_idxs:
                aligned_asr_idxs.extend(gt2asr.get(wi, []))
            aligned_asr_idxs = sorted(set(aligned_asr_idxs))

            if aligned_asr_idxs:
               
                j_lo, j_hi = min(aligned_asr_idxs), max(aligned_asr_idxs)
                window = asr_words[j_lo:j_hi+1]

                # Exact-match search restricted to the aligned window
                target_tokens = entity_norm.split()
                n = len(target_tokens)
                for k in range(0, len(window) - n + 1):
                    ngram = ' '.join(window[k:k+n])
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


all_log = pd.DataFrame(entity_leakage_records)
all_log.to_csv("entityleakagereportrate50.csv", index=False)

leaked_only = all_log[all_log["leaked"] == True]
leaked_only.to_csv("leaked_entitiesrate50.csv", index=False)

#Entity-based leakage percentage
total_entities = len(all_log)
leaked_entities = int(all_log["leaked"].sum())  
overall_leak_percent = (leaked_entities / total_entities * 100) if total_entities else 0.0

print("\n===  Entity-Based Leakage (Overall) ===")
print(f" Total entities (GT) : {total_entities}")
print(f" Leaked entities     : {leaked_entities}")
print(f"Leakage Percent     : {overall_leak_percent:.2f}%")


per_clip = (
    all_log.groupby(["match_id", "audio_filename"])["leaked"]
           .agg(total_entities_in_clip="count", leaked_entities_in_clip="sum")
           .reset_index()
)


per_clip.to_csv("clip_leakage_entity_based_rate50.csv", index=False)


weighted_leak_percent = (
    per_clip["leaked_entities_in_clip"].sum() / per_clip["total_entities_in_clip"].sum() * 100
    if per_clip["total_entities_in_clip"].sum() > 0 else 0.0
)

print("\n===  Weighted (Global) Leakage Rate ===")
print(f" Weighted Leakage Percent : {weighted_leak_percent:.2f}%")


TP = leaked_entities
FN = total_entities - leaked_entities

precision = 1.0  # evaluating only against GT entities
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n===  NER Leakage Evaluation (Recall-focused) ===")
print(f" True Positives (leaked)   : {TP}")
print(f"False Negatives (masked)  : {FN}")
print(f"Precision (assumed 1.0)   : {precision:.2f}")
print(f"Recall (leakage ratio)    : {recall:.4f}")
print(f"F1-Score                  : {f1:.4f}")
