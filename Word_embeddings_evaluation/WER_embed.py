

"""
CSV ➜ WER / WER-E / WER-S scorer (FastText cc.xx.300.bin)

INPUTS
- reference CSV with columns:
    - audio file column (e.g., "audio" or "filename")
    - transcript column (e.g., "text" or "transcript")
- hypothesis CSV with (possibly different) audio file naming, e.g.:
    ref: clip_0267_speaker_204733.wav
    hyp: clip_0267_speaker_204733_with_246_snr15.wav
  The script normalizes both to a common key: "clip_0267_speaker_204733".

WHAT IT DOES
- Joins ref/hyp rows by that common key
- Computes Classic WER, WER-E, WER-S per utterance and corpus averages
- Saves a CSV report

USAGE
- Install once:  pip install gensim==4.3.2 pandas
- Put your files/paths in the CONFIG section below
- Run:  python wer_from_csv.py
"""

import re
import os
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from gensim.models.fasttext import load_facebook_vectors

# =========================
# ======== CONFIG =========
# =========================
# FastText model 
FASTTEXT_BIN = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/fastText/cc.en.300.bin"

# Reference CSV
REF_CSV = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull/all_transcriptsnw.csv"
REF_AUDIO_COL = "audio_filename"       
REF_TEXT_COL  = "normalized_text"  

# Hypothesis CSV
HYP_CSV = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1NewsampledTranscriptrate40/Sampledrate40.csv"
HYP_AUDIO_COL = "audio_filename"       
HYP_TEXT_COL  = "transcript"  

# Output
OUTPUT_CSV = "wer_report_rate40.csv"

# Optional: set to True to also write alignment op-path strings
WRITE_ALIGNMENT = False


TOKEN_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall((s or "").lower())


KEY_RE = re.compile(r"(clip_\d+_speaker_\d+)", re.IGNORECASE)

def audio_key(filename: str) -> str:
    if not filename:
        return ""
    base = os.path.basename(filename)
    m = KEY_RE.search(base)
    if m:
        return m.group(1).lower()
    # fallback: strip extension and return basename
    return os.path.splitext(base)[0].lower()

# =========================
# === FASTTEXT WRAPPER ====
# =========================
class FT:
    def __init__(self, path: str):
        self.kv = load_facebook_vectors(path)
        self.dim = self.kv.vector_size
    def vec(self, w: str) -> np.ndarray:
        # FastTextKeyedVectors composes OOV via subwords
        return self.kv.get_vector(w)

def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 1.0
    sim = float(np.dot(u, v) / (nu * nv))
    # clamp for numerical safety
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim   # in [0,2]

# =========================
# === ALIGNMENT (DP) ======
# =========================
def align_classic(ref: List[str], hyp: List[str]):
    """
    Standard Levenshtein with unit costs.
    Returns:
      path:  [(op, r_tok, h_tok)]  op ∈ {"MATCH","SUB","INS","DEL"}
      counts: (S, I, D)
    """
    N, M = len(ref), len(hyp)
    dp = [[0]*(M+1) for _ in range(N+1)]
    back = [[None]*(M+1) for _ in range(N+1)]

    for i in range(1, N+1):
        dp[i][0] = i
        back[i][0] = ('DEL', i-1, 0)
    for j in range(1, M+1):
        dp[0][j] = j
        back[0][j] = ('INS', 0, j-1)

    for i in range(1, N+1):
        ri = ref[i-1]
        for j in range(1, M+1):
            hj = hyp[j-1]
            sub_cost = 0 if ri == hj else 1
            cand = [
                (dp[i-1][j] + 1, ('DEL', i-1, j)),
                (dp[i][j-1] + 1, ('INS', i, j-1)),
                (dp[i-1][j-1] + sub_cost, ('MATCH' if sub_cost == 0 else 'SUB', i-1, j-1)),
            ]
            dp[i][j], back[i][j] = min(cand, key=lambda x: x[0])

    path = []
    S = I = D = 0
    i, j = N, M
    while i > 0 or j > 0:
        op, pi, pj = back[i][j]
        if op in ('MATCH', 'SUB'):
            if op == 'SUB': S += 1
            path.append((op, ref[pi], hyp[pj]))
        elif op == 'DEL':
            D += 1
            path.append(('DEL', ref[pi], '∅'))
        else:
            I += 1
            path.append(('INS', '∅', hyp[pj]))
        i, j = pi, pj
    path.reverse()
    return path, (S, I, D)

def classic_wer(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    _, (S, I, D) = align_classic(ref_tokens, hyp_tokens)
    N = max(1, len(ref_tokens))
    return (S + I + D) / N

def wer_e(ref_tokens: List[str], hyp_tokens: List[str], ft: FT) -> float:
    path, _ = align_classic(ref_tokens, hyp_tokens)
    total = 0.0
    for op, r, h in path:
        if op == 'MATCH':
            continue
        elif op == 'SUB':
            total += cosine_distance(ft.vec(r), ft.vec(h))
        else:
            total += 1.0
    return total / max(1, len(ref_tokens))

def wer_s(ref_tokens: List[str], hyp_tokens: List[str], ft: FT) -> float:
    N, M = len(ref_tokens), len(hyp_tokens)
    vref = [ft.vec(w) for w in ref_tokens]
    vhyp = [ft.vec(w) for w in hyp_tokens]

    dp = np.zeros((N+1, M+1), dtype=np.float32)
    for i in range(1, N+1): dp[i,0] = i
    for j in range(1, M+1): dp[0,j] = j

    for i in range(1, N+1):
        for j in range(1, M+1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                sub_cost = 0.0
            else:
                sub_cost = cosine_distance(vref[i-1], vhyp[j-1])
            dp[i,j] = min(
                dp[i-1,j] + 1.0,        # deletion
                dp[i,j-1] + 1.0,        # insertion
                dp[i-1,j-1] + sub_cost  # substitution / match
            )
    return float(dp[N,M]) / max(1, N)

def path_to_string(path) -> str:
    # Optional pretty path
    parts = []
    for op, r, h in path:
        if op == "MATCH":
            parts.append(f"=({r})")
        elif op == "SUB":
            parts.append(f"S({r}->{h})")
        elif op == "DEL":
            parts.append(f"D({r})")
        else:
            parts.append(f"I({h})")
    return " ".join(parts)

# =========================
# ====== PIPELINE =========
# =========================
def load_csv_as_map(csv_path: str, audio_col: str, text_col: str) -> Dict[str, Tuple[str, str]]:
    """
    Returns: key -> (audio_filename, transcript)
    """
    df = pd.read_csv(csv_path)
    if audio_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"{csv_path} must contain columns '{audio_col}' and '{text_col}'")
    out = {}
    for _, row in df.iterrows():
        a = str(row[audio_col]) if not pd.isna(row[audio_col]) else ""
        t = str(row[text_col]) if not pd.isna(row[text_col]) else ""
        k = audio_key(a)
        if not k:
            continue
       
        out[k] = (a, t)
    return out

def main():
    print("Loading FastText model…")
    ft = FT(FASTTEXT_BIN)

    ref_map = load_csv_as_map(REF_CSV, REF_AUDIO_COL, REF_TEXT_COL)
    hyp_map = load_csv_as_map(HYP_CSV, HYP_AUDIO_COL, HYP_TEXT_COL)

    keys = sorted(set(ref_map.keys()) & set(hyp_map.keys()))
   

   
    rows = []
    wer_vals, were_vals, wers_vals = [], [], []

    for k in keys:
        ref_audio, ref_txt = ref_map[k]
        hyp_audio, hyp_txt = hyp_map[k]

        rtoks = tokenize(ref_txt)
        htoks = tokenize(hyp_txt)

        # Classic WER and alignment
        path, (S, I, D) = align_classic(rtoks, htoks)
        w  = (S + I + D) / max(1, len(rtoks))

        # WER-E (same path, soft substitutions)
        total = 0.0
        for op, r, h in path:
            if op == "MATCH":
                continue
            elif op == "SUB":
                total += cosine_distance(ft.vec(r), ft.vec(h))
            else:
                total += 1.0
        we = total / max(1, len(rtoks))

        # WER-S (soft DP)
        ws = wer_s(rtoks, htoks, ft)

        wer_vals.append(w); were_vals.append(we); wers_vals.append(ws)

        row = {
            "key": k,
            "ref_audio": ref_audio,
            "hyp_audio": hyp_audio,
            "ref_len": len(rtoks),
            "S": S, "I": I, "D": D,
            "WER": round(w, 6),
            "WER_E": round(we, 6),
            "WER_S": round(ws, 6),
        }
        if WRITE_ALIGNMENT:
            row["alignment"] = path_to_string(path)
        rows.append(row)

    # Save CSV
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-utterance report saved to: {os.path.abspath(OUTPUT_CSV)}")

    # Corpus averages
    def avg(a): return float(np.mean(a)) if a else float("nan")
    print("\n=== Corpus-level ===")
    print(f"Pairs evaluated : {len(keys)}")
    print(f"WER   = {avg(wer_vals):.6f}")
    print(f"WER-E = {avg(were_vals):.6f}")
    print(f"WER-S = {avg(wers_vals):.6f}")

   
if __name__ == "__main__":
    main()
