

"""
Classic → Entities → Embeddings order + Embedding-space entity search.

Evaluations per utterance:
1) Classic WER (classic path)
2) Entity WER (classic path)
3) WER-E (classic path, SUB = 1 - cosine)
4) Entity WER-E (classic path)
5) WER-S (soft DP path, SUB = 1 - cosine)
6) Entity WER-S (soft DP path)


Outputs:
- Per-utterance CSV: wer_report_rate30_entities.csv
- Per-entity CSV   : entity_errors_report.csv
  Includes:
    - entity_text_ref
    - entity_text_hyp (soft-aligned)
    - entity_soft_aligned_span_cosine
    - entity_best_hyp_span, entity_best_span_len, entity_best_span_cosine
    - classic/wer_e/wer_s rates & status + op counts
"""

import os, re, ast
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from gensim.models.fasttext import load_facebook_vectors


FASTTEXT_BIN = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/fastText/cc.en.300.bin"


REF_CSV = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull/all_transcriptsnw.csv"
REF_AUDIO_COL = "audio_filename"
REF_TEXT_COL  = "normalized_text"
REF_ENTITY_COL = "normalized_ner"  

# Hypothesis CSV
HYP_CSV = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1NewsampledTranscriptrate50/Sampledrate50.csv"
HYP_AUDIO_COL = "audio_filename"
HYP_TEXT_COL  = "transcript"

# Outputs
OUTPUT_PER_UTT_CSV   = "wer_report_rate50_entitiesnw.csv"
OUTPUT_ENTITIES_CSV  = "entity_errors_reportrate50nw.csv"


COUNT_INS_AS_ENTITY = False

# Embedding-space search window (max hyp span length to consider)
MAX_HYP_SPAN_LEN = 6  


TOKEN_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)
def tokenize_with_spans(s: str):
    s = s or ""
    toks, spans = [], []
    for m in TOKEN_RE.finditer(s.lower()):
        toks.append(m.group(0))
        spans.append((m.start(), m.end()))
    return toks, spans

KEY_RE = re.compile(r"(clip_\d+_speaker_\d+)", re.IGNORECASE)
def audio_key(filename: str) -> str:
    if not filename:
        return ""
    base = os.path.basename(filename)
    m = KEY_RE.search(base)
    if m:
        return m.group(1).lower()
    return os.path.splitext(base)[0].lower()


class FT:
    def __init__(self, path: str):#fastext
        self.kv = load_facebook_vectors(path)
        self.dim = self.kv.vector_size
    def vec(self, w: str) -> np.ndarray:
        return self.kv.get_vector(w)

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))

def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    # distance used by WER-E/WER-S substitution costs
    sim = cosine_similarity(u, v)
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim  # [0,2]

ENT_TRIPLE_RE = re.compile(r"\[\s*'([^']+)'\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")

def parse_entity_list(cell: Any) -> List[Tuple[str, int, int]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) >= 3 for x in obj):
            return [(str(x[0]), int(x[1]), int(x[2])) for x in obj]
    except Exception:
        pass
    out = []
    for m in ENT_TRIPLE_RE.finditer(s):
        out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return out

def entity_token_mask(token_spans: List[Tuple[int,int]], entities: List[Tuple[str,int,int]]) -> List[bool]:
    spans = [(s, s+l) for (_lab, s, l) in entities]
    mask = []
    for (a,b) in token_spans:
        mask.append(any(not (b <= es or a >= ee) for (es,ee) in spans))
    return mask

def span_text(full_text: str, start: int, length: int) -> str:
    end = start + length
    if start < 0 or end > len(full_text): return ""
    return full_text[start:end]
#alignment
def align_classic(ref_toks: List[str], hyp_toks: List[str]):
    """Return (path, (S,I,D)), path items: (op, i_ref, r_tok, j_hyp, h_tok)"""
    N, M = len(ref_toks), len(hyp_toks)
    dp = [[0]*(M+1) for _ in range(N+1)]
    back = [[None]*(M+1) for _ in range(N+1)]
    for i in range(1, N+1):
        dp[i][0] = i; back[i][0] = ('DEL', i-1, None)
    for j in range(1, M+1):
        dp[0][j] = j; back[0][j] = ('INS', None, j-1)
    for i in range(1, N+1):
        ri = ref_toks[i-1]
        for j in range(1, M+1):
            hj = hyp_toks[j-1]
            sub_cost = 0 if ri == hj else 1
            cand = [
                (dp[i-1][j] + 1, ('DEL', i-1, None)),
                (dp[i][j-1] + 1, ('INS', None, j-1)),
                (dp[i-1][j-1] + sub_cost, ('MATCH' if sub_cost==0 else 'SUB', i-1, j-1)),
            ]
            dp[i][j], back[i][j] = min(cand, key=lambda x: x[0])
    path = []
    S=I=D=0; i,j=N,M
    while i>0 or j>0:
        op, ii, jj = back[i][j]
        if op=='MATCH':
            path.append((op, ii, ref_toks[ii], jj, hyp_toks[jj])); i,j=ii,jj
        elif op=='SUB':
            S+=1; path.append((op, ii, ref_toks[ii], jj, hyp_toks[jj])); i,j=ii,jj
        elif op=='DEL':
            D+=1; path.append((op, ii, ref_toks[ii], None, None)); i,j=ii,j
        else:
            I+=1; path.append((op, None, None, jj, hyp_toks[jj])); i,j=i,jj
    path.reverse()
    return path, (S,I,D)

def soft_align(ref_toks, hyp_toks, vref, vhyp):
    """Soft DP path (WER-S), path items: (op, i_ref, r_tok, j_hyp, h_tok)"""
    N, M = len(ref_toks), len(hyp_toks)
    dp = np.zeros((N+1, M+1), dtype=np.float32)
    back: List[List[Optional[Tuple[str, Optional[int], Optional[int]]]]] = [[None]*(M+1) for _ in range(N+1)]
    for i in range(1, N+1):
        dp[i,0]=i; back[i][0]=('DEL', i-1, None)
    for j in range(1, M+1):
        dp[0,j]=j; back[0][j]=('INS', None, j-1)
    for i in range(1, N+1):
        ri = ref_toks[i-1]
        for j in range(1, M+1):
            hj = hyp_toks[j-1]
            sub_cost = 0.0 if ri==hj else cosine_distance(vref[i-1], vhyp[j-1])
            cand = [
                (dp[i-1,j] + 1.0, ('DEL', i-1, None)),
                (dp[i,j-1] + 1.0, ('INS', None, j-1)),
                (dp[i-1,j-1] + sub_cost, ('MATCH' if sub_cost==0.0 else 'SUB', i-1, j-1))
            ]
            dp[i,j], back[i][j] = min(cand, key=lambda x: x[0])
    path = []
    i,j=N,M
    while i>0 or j>0:
        op, ii, jj = back[i][j]
        if op in ('MATCH','SUB'):
            path.append((op, ii, ref_toks[ii], jj, hyp_toks[jj])); i,j=ii,jj
        elif op=='DEL':
            path.append((op, ii, ref_toks[ii], None, None)); i,j=ii,j
        else:
            path.append((op, None, None, jj, hyp_toks[jj])); i,j=i,jj
    path.reverse()
    return path

def classic_wer_from_path(path, ref_len: int) -> float:
    S=I=D=0
    for (op, *_rest) in path:
        if op=='SUB': S+=1
        elif op=='DEL': D+=1
        elif op=='INS': I+=1
    return (S+I+D)/max(1, ref_len)

def wer_e_from_path(path, ft: FT, ref_len: int) -> float:
    total=0.0
    for (op, i_ref, r_tok, j_hyp, h_tok) in path:
        if op=='MATCH': continue
        elif op=='SUB': total += cosine_distance(ft.vec(r_tok), ft.vec(h_tok))
        else: total += 1.0
    return total/max(1, ref_len)

def entity_rates_from_path(path,
                           ref_entity_mask: List[bool],
                           ft: Optional[FT],
                           ref_ent_len: int,
                           count_ins_as_entity: bool,
                           soft_cost: bool):
    """
    Return (rate, subs, dels, ins) restricted to tokens where ref_entity_mask=True.
    - soft_cost=True => use cosine for SUB; else 1.0.
    """
    if ref_ent_len <= 0:
        return (np.nan, 0, 0, 0)
    total = 0.0
    subs=dels=ins=0
    last_entity_idx = None
    for (op, i_ref, r_tok, j_hyp, h_tok) in path:
        if op=='SUB':
            if i_ref is not None and ref_entity_mask[i_ref]:
                subs += 1
                if soft_cost and ft is not None:
                    total += cosine_distance(ft.vec(r_tok), ft.vec(h_tok))
                else:
                    total += 1.0
            last_entity_idx = i_ref
        elif op=='DEL':
            if i_ref is not None and ref_entity_mask[i_ref]:
                dels += 1
                total += 1.0
            last_entity_idx = i_ref
        elif op=='MATCH':
            last_entity_idx = i_ref
        elif op=='INS':
            if count_ins_as_entity:
                if last_entity_idx is not None and last_entity_idx>=0 and ref_entity_mask[last_entity_idx]:
                    ins += 1
                    total += 1.0
    return (total/max(1, ref_ent_len), subs, dels, ins)


def mean_vec(vecs: List[np.ndarray], idxs: List[int]) -> np.ndarray:
    if not idxs:
        return np.zeros_like(vecs[0]) if vecs else np.zeros(300, dtype=np.float32)
    s = np.sum([vecs[i] for i in idxs], axis=0)
    return s / float(len(idxs))

def best_span_by_cosine(ref_vec: np.ndarray,
                        hyp_tokens: List[str],
                        hyp_vecs: List[np.ndarray],
                        max_span_len: int = 6):
    """
    Brute-force search over all hyp spans up to length max_span_len.
    Returns (best_start, best_len, best_sim, best_text)
    """
    M = len(hyp_tokens)
    if M == 0:
        return (-1, 0, 0.0, "")
    best = (-1, 0, -1.0, "")
    # Prefix sums for fast mean vectors
    cums = np.zeros((M+1, hyp_vecs[0].shape[0]), dtype=np.float32)
    for i in range(M):
        cums[i+1] = cums[i] + hyp_vecs[i]
    maxL = min(max_span_len, M)
    for L in range(1, maxL+1):
        for s in range(0, M-L+1):
            span_sum = cums[s+L] - cums[s]
            span_mean = span_sum / float(L)
            sim = cosine_similarity(ref_vec, span_mean)
            if sim > best[2]:
                best = (s, L, sim, " ".join(hyp_tokens[s:s+L]))
    return best


def load_csv_as_map(csv_path: str, audio_col: str, text_col: str,
                    ent_col: Optional[str] = None):
    df = pd.read_csv(csv_path)
    if audio_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"{csv_path} must contain '{audio_col}' and '{text_col}'")
    has_ent = ent_col and (ent_col in df.columns)
    out = {}
    for _, row in df.iterrows():
        a = "" if pd.isna(row[audio_col]) else str(row[audio_col])
        t = "" if pd.isna(row[text_col]) else str(row[text_col])
        k = audio_key(a)
        if not k:
            continue
        toks, spans = tokenize_with_spans(t)
        entry = {
            "audio": a, "text": t, "tokens": toks, "spans": spans,
            "entities": [], "entity_mask": [False]*len(toks), "n_ent_tokens": 0
        }
        if has_ent:
            ents = parse_entity_list(row[ent_col])
            entry["entities"] = ents
            mask = entity_token_mask(spans, ents)
            entry["entity_mask"] = mask
            entry["n_ent_tokens"] = int(sum(1 for b in mask if b))
        out[k] = entry
    return out

def mask_from_idxs(n: int, idxs: List[int]) -> List[bool]:
    m = [False]*n
    for i in idxs:
        if 0 <= i < n: m[i] = True
    return m

#load fastext model first, download it and save in directory
def main():
    print("Loading FastText model …")
    ft = FT(FASTTEXT_BIN)

    print("Reading CSVs …")
    ref_map = load_csv_as_map(REF_CSV, REF_AUDIO_COL, REF_TEXT_COL, REF_ENTITY_COL)
    hyp_map = load_csv_as_map(HYP_CSV, HYP_AUDIO_COL, HYP_TEXT_COL, ent_col=None)

    keys = sorted(set(ref_map.keys()) & set(hyp_map.keys()))
    print(f"Matched pairs: {len(keys)}")

    per_utt_rows = []
    ent_rows = []

    
    wer_vals=[]; were_vals=[]; wers_vals=[]
    wer_ent_vals=[]; were_ent_vals=[]; wers_ent_vals=[]
    ent_cos_vals=[]   

    for k in keys:
        ref = ref_map[k]; hyp = hyp_map[k]
        rtoks = ref["tokens"]; htoks = hyp["tokens"]

        vref_all = [ft.vec(w) for w in rtoks]
        vhyp_all = [ft.vec(w) for w in htoks]

        # Classic path
        path_cls, (S,I,D) = align_classic(rtoks, htoks)
        w_classic = classic_wer_from_path(path_cls, len(rtoks))

        #  Entity WER (classic path)
        ent_mask = ref["entity_mask"]
        n_ent = ref["n_ent_tokens"]
        if n_ent > 0:
            rate_ent_cls, *_ = entity_rates_from_path(
                path_cls, ent_mask, ft=None, ref_ent_len=n_ent,
                count_ins_as_entity=COUNT_INS_AS_ENTITY, soft_cost=False
            )
        else:
            rate_ent_cls = np.nan

        #  WER-E (classic path)
        w_wer_e = wer_e_from_path(path_cls, ft, len(rtoks))

        # Entity WER-E (classic path) 
        if n_ent > 0:
            rate_ent_e, *_ = entity_rates_from_path(
                path_cls, ent_mask, ft=ft, ref_ent_len=n_ent,
                count_ins_as_entity=COUNT_INS_AS_ENTITY, soft_cost=True
            )
        else:
            rate_ent_e = np.nan

        #  Soft path (WER-S)
        path_soft = soft_align(rtoks, htoks, vref_all, vhyp_all)
        w_soft = wer_e_from_path(path_soft, ft, len(rtoks)) 

        #  Entity WER-S (soft path)
        if n_ent > 0:
            rate_ent_s, *_ = entity_rates_from_path(
                path_soft, ent_mask, ft=ft, ref_ent_len=n_ent,
                count_ins_as_entity=COUNT_INS_AS_ENTITY, soft_cost=True
            )
        else:
            rate_ent_s = np.nan

        
        if ref["entities"]:
            for (lab, start, length) in ref["entities"]:
               
                token_idxs = [ti for ti,(a,b) in enumerate(ref["spans"])
                              if not (b <= start or a >= start+length)]
                if not token_idxs:
                    continue
                ent_text_ref = span_text(ref["text"], start, length)
                span_mask = mask_from_idxs(len(rtoks), token_idxs)

                # Classic-path entity
                r_cls, s1,d1,i1 = entity_rates_from_path(
                    path_cls, span_mask, ft=None, ref_ent_len=len(token_idxs),
                    count_ins_as_entity=COUNT_INS_AS_ENTITY, soft_cost=False
                )
                # WER-E entity (classic path)
                r_e,   s2,d2,i2 = entity_rates_from_path(
                    path_cls, span_mask, ft=ft, ref_ent_len=len(token_idxs),
                    count_ins_as_entity=COUNT_INS_AS_ENTITY, soft_cost=True
                )
                # WER-S entity (soft path)
                r_s,   s3,d3,i3 = entity_rates_from_path(
                    path_soft, span_mask, ft=ft, ref_ent_len=len(token_idxs),
                    count_ins_as_entity=COUNT_INS_AS_ENTITY, soft_cost=True
                )

                
                ref_entity_vec = mean_vec(vref_all, token_idxs)

            
                soft_hyp_tokens = []
                soft_hyp_idxs = []
                for (op, i_ref, r_tok, j_hyp, h_tok) in path_soft:
                    if i_ref in token_idxs:
                        if j_hyp is not None and h_tok:
                            soft_hyp_tokens.append(h_tok)
                            soft_hyp_idxs.append(j_hyp)
                soft_aligned_text = " ".join(soft_hyp_tokens)
                if soft_hyp_idxs:
                    soft_span_vec = mean_vec(vhyp_all, soft_hyp_idxs)
                    soft_aligned_cos = cosine_similarity(ref_entity_vec, soft_span_vec)
                    ent_cos_vals.append(soft_aligned_cos)  
                else:
                    soft_aligned_cos = float("nan")

                best_s, best_L, best_sim, best_text = best_span_by_cosine(
                    ref_entity_vec, htoks, vhyp_all, max_span_len=MAX_HYP_SPAN_LEN
                )

                ent_rows.append({
                    "key": k,
                    "label": lab,
                    "start": start,
                    "length": length,
                    "entity_text_ref": ent_text_ref,
                    "entity_text_hyp": soft_aligned_text,       # from SOFT alignment
                    "entity_soft_aligned_span_cosine": (round(soft_aligned_cos,6) if soft_hyp_idxs else ""),
                    "entity_best_hyp_span": best_text,          # from SEARCH
                    "entity_best_span_len": best_L,
                    "entity_best_span_cosine": round(best_sim,6),
                    "n_tokens": len(token_idxs),
                    "classic_rate": round(r_cls,6) if not np.isnan(r_cls) else "",
                    "wer_e_rate":   round(r_e,6)   if not np.isnan(r_e)   else "",
                    "wer_s_rate":   round(r_s,6)   if not np.isnan(r_s)   else "",
                    "status_classic": ("OK" if (not np.isnan(r_cls) and r_cls==0.0) else "ERR"),
                    "status_wer_e":  ("OK" if (not np.isnan(r_e)   and r_e==0.0)   else "ERR"),
                    "status_wer_s":  ("OK" if (not np.isnan(r_s)   and r_s==0.0)   else "ERR"),
                    "subs_cls": s1, "dels_cls": d1, "ins_cls": i1,
                    "subs_e":   s2, "dels_e":  d2, "ins_e":  i2,
                    "subs_s":   s3, "dels_s":  d3, "ins_s":  i3,
                })

       
        per_utt_rows.append({
            "key": k,
            "ref_audio": ref["audio"],
            "hyp_audio": hyp["audio"],
            "ref_len": len(rtoks),
            "S": S, "I": I, "D": D,
            "WER": round(w_classic, 6),
            "WER_E": round(w_wer_e, 6),
            "WER_S": round(w_soft, 6),
            "ent_tokens": n_ent,
            "WER_ENT": (round(rate_ent_cls, 6) if not np.isnan(rate_ent_cls) else ""),
            "WER_E_ENT": (round(rate_ent_e, 6) if not np.isnan(rate_ent_e) else ""),
            "WER_S_ENT": (round(rate_ent_s, 6) if not np.isnan(rate_ent_s) else ""),
        })

        
        wer_vals.append(w_classic); were_vals.append(w_wer_e); wers_vals.append(w_soft)
        if n_ent > 0 and not np.isnan(rate_ent_cls):
            wer_ent_vals.append(rate_ent_cls)
            were_ent_vals.append(rate_ent_e)
            wers_ent_vals.append(rate_ent_s)

    
    pd.DataFrame(per_utt_rows).to_csv(OUTPUT_PER_UTT_CSV, index=False)
    print(f"Per-utterance report → {os.path.abspath(OUTPUT_PER_UTT_CSV)}")
    if ent_rows:
        pd.DataFrame(ent_rows).to_csv(OUTPUT_ENTITIES_CSV, index=False)
        print(f"Per-entity report   → {os.path.abspath(OUTPUT_ENTITIES_CSV)}")
    else:
        print("No entities found — per-entity CSV not written.")

    
    def avg(a): return float(np.nanmean(a)) if a else float("nan")
    print("\n=== Corpus-level (macro averages) ===")
    print(f"Pairs evaluated : {len(per_utt_rows)}")
    print(f"WER        = {avg(wer_vals):.6f}")
    print(f"WER-E      = {avg(were_vals):.6f}")
    print(f"WER-S      = {avg(wers_vals):.6f}")
    if wer_ent_vals:
        print(f"WER_ENT    = {avg(wer_ent_vals):.6f}")
        print(f"WER-E_ENT  = {avg(were_ent_vals):.6f}")
        print(f"WER-S_ENT  = {avg(wers_ent_vals):.6f}")

     ##Avg cosine similarity 
    if ent_cos_vals:
        avg_ent_cos = float(np.nanmean(ent_cos_vals))
        print(f"Avg entity soft-aligned cosine = {avg_ent_cos:.6f}")

if __name__ == "__main__":
    main()
