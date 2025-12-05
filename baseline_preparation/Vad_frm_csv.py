import os
import re
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import webrtcvad

# ================= helpers =================

def syllable_count(text: str) -> int:
    """Simple English-ish heuristic (ok for relative SPS)."""
    if not isinstance(text, str) or not text:
        return 0
    return len(re.findall(r'[aeiouy]+', text.lower()))

def to_pcm16_mono_16k(path: str, target_sr: int = 16000):
    """
    Load any audio (ogg/wav/mp3/flac), convert -> mono 16k.
    Returns (sr, duration_sec, pcm_int16_array).
    """
    # Always get float32 in [-1, 1]
    y, sr = sf.read(path, dtype='float32', always_2d=False)

    # Downmix to mono if needed (take first channel)
    if hasattr(y, "ndim") and y.ndim > 1:
        y = y[:, 0]

    # Resample if needed (librosa expects float)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Clip to [-1, 1], convert to int16 PCM
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    duration_sec = float(len(y)) / sr if len(y) else 0.0
    pcm16 = (y * 32767.0).astype(np.int16)  # use 32767 to avoid overflow
    return sr, duration_sec, pcm16

def frame_generator_int16(pcm16: np.ndarray, sr: int, frame_ms: int = 30):
    """
    Slice an int16 mono signal into equal frames (10/20/30 ms).
    Returns (frames_int16_array [n_frames, frame_len], frame_len).
    """
    assert frame_ms in (10, 20, 30), "webrtcvad requires frame_ms in {10,20,30}"
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0:
        return np.empty((0, 0), dtype=np.int16), 0
    n_frames = pcm16.size // frame_len
    if n_frames == 0:
        return np.empty((0, frame_len), dtype=np.int16), frame_len
    trimmed = pcm16[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)
    return frames, frame_len

def vad_flags_from_frames(frames: np.ndarray, sr: int, frame_ms: int = 30, mode: int = 2) -> np.ndarray:
    """
    Run WebRTC VAD on int16 frames, return boolean flags per frame.
    """
    vad = webrtcvad.Vad(mode)
    if frames.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.array([vad.is_speech(fr.tobytes(), sr) for fr in frames], dtype=bool)

def rms_int16(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(x.astype(np.float64)**2)))

def dbfs_from_rms_int16(rms_val: float) -> float:
    if np.isnan(rms_val) or rms_val <= 0:
        return float('nan')
    # Full-scale reference for int16 is 32768
    return 20.0 * math.log10(rms_val / 32768.0)

# ================= column resolver =================

ALIASES = {
    "speaker_id": ["speaker_id", "spk", "spk_id", "speaker"],
    "audio_filename": ["audio_filename", "audio", "file", "filename", "path", "wav", "ogg"],
    "reference_transcript": ["reference_transcript", "reference", "ref", "reference_text", "text", "transcript"],
    "hypothesis_transcript": ["hypothesis_transcript", "hypothesis", "hyp", "prediction", "pred", "asr"],
    "WER": ["WER", "wer", "word_error_rate"],
}

def build_colmap(df: pd.DataFrame) -> dict:
    lower_to_real = {c.lower(): c for c in df.columns}
    resolved = {}
    missing_keys = []
    for canonical, options in ALIASES.items():
        found = None
        for opt in options:
            if opt.lower() in lower_to_real:
                found = lower_to_real[opt.lower()]
                break
        if found is not None:
            resolved[canonical] = found
        else:
            if canonical == "audio_filename":
                missing_keys.append(canonical)
    if missing_keys:
        raise ValueError(
            "CSV missing required column for audio path.\n"
            f"Accepted aliases for audio file: {ALIASES['audio_filename']}\n"
            f"Found headers: {list(df.columns)}"
        )
    return resolved

# ================= main =================

def analyze_audio(csv_path: str, audio_dir: str, frame_ms: int = 30, vad_mode: int = 2) -> pd.DataFrame:
    """
    Simple pipeline:
    - reads CSV (expects at least an audio filename column)
    - loads audio from audio_dir (supports .ogg/.wav/.mp3/.flac)
    - runs VAD to split voiced/unvoiced frames
    - computes:
        * WPM and SPS from reference transcript (if present)
        * noise_rms / noise_dbfs from unvoiced frames (if enough unvoiced)
        * speech_rms and SNR (if both RMS available)
        * noise_estimate_reliable flag (unvoiced_sec >= 0.3 s)
    - passes through WER and speaker_id if present
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] CSV has no rows: {csv_path}")
        return pd.DataFrame()

    colmap = build_colmap(df)

    out_rows = []
    for _, row in df.iterrows():
        
        audio_file = str(row[colmap["audio_filename"]])
        audio_path = audio_file if os.path.isabs(audio_file) else os.path.join(audio_dir, audio_file)
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio: {audio_path}")
            continue

      
        spk = row[colmap["speaker_id"]] if "speaker_id" in colmap and not pd.isna(row[colmap["speaker_id"]]) else None
        ref_text = row[colmap["reference_transcript"]] if "reference_transcript" in colmap else ""
        ref_text = "" if (ref_text is None or (isinstance(ref_text, float) and pd.isna(ref_text))) else str(ref_text)
        wer_val = None
        if "WER" in colmap and not pd.isna(row[colmap["WER"]]):
            try:
                wer_val = float(row[colmap["WER"]])
            except Exception:
                wer_val = None

        try:
            # Load & prepare audio
            sr, duration_sec, pcm16 = to_pcm16_mono_16k(audio_path, target_sr=16000)

            # Frame & VAD
            frames, frame_len = frame_generator_int16(pcm16, sr=sr, frame_ms=frame_ms)
            flags = vad_flags_from_frames(frames, sr=sr, frame_ms=frame_ms, mode=vad_mode)

            n_frames = int(flags.size)
            speech_frames = int(flags.sum())
            unvoiced_frames = n_frames - speech_frames
            secs_per_frame = (frame_len / sr) if (sr > 0 and frame_len > 0) else 0.0
            speech_sec = speech_frames * secs_per_frame
            unvoiced_sec = unvoiced_frames * secs_per_frame
            speech_ratio = (speech_frames / n_frames) if n_frames > 0 else 0.0

            # Collect samples for noise (unvoiced) and speech (voiced)
            noise_samples = frames[~flags].reshape(-1) if unvoiced_frames > 0 else np.array([], dtype=np.int16)
            speech_samples = frames[flags].reshape(-1) if speech_frames > 0 else np.array([], dtype=np.int16)

            # Noise estimation reliability
            noise_estimate_reliable = unvoiced_sec >= 0.30  # ≥300 ms of silence

            # RMS / dBFS / SNR
            noise_rms = rms_int16(noise_samples) if noise_estimate_reliable else float('nan')
            noise_dbfs = dbfs_from_rms_int16(noise_rms) if noise_estimate_reliable else float('nan')
            speech_rms = rms_int16(speech_samples) if speech_samples.size else float('nan')

            snr_db = None
            if (not np.isnan(noise_rms)) and (noise_rms > 0) and (not np.isnan(speech_rms)) and (speech_rms > 0):
                snr_db = round(20.0 * math.log10(speech_rms / noise_rms), 2)

            # Fluency (from reference transcript if available)
            words = len(ref_text.split()) if ref_text else 0
            syllables = syllable_count(ref_text) if ref_text else 0
            wpm = (words / duration_sec) * 60.0 if duration_sec > 0 else None
            sps = (syllables / duration_sec) if duration_sec > 0 else None

            out = {
                "speaker_id": spk,
                "audio_filename": os.path.basename(audio_path),
                "duration_sec": round(duration_sec, 3),
                "num_frames": n_frames,
                "speech_frames": speech_frames,
                "unvoiced_frames": unvoiced_frames,
                "speech_ratio": round(speech_ratio, 4),
                "speech_sec": round(speech_sec, 3),
                "unvoiced_sec": round(unvoiced_sec, 3),

                # fluency
                "words_per_minute": None if wpm is None else round(wpm, 2),
                "syllables_per_second": None if sps is None else round(sps, 2),

                # noise / SNR
                "noise_estimate_reliable": bool(noise_estimate_reliable),
                "noise_rms": None if np.isnan(noise_rms) else round(float(noise_rms), 6),
                "noise_dbfs": None if (np.isnan(noise_dbfs) or np.isneginf(noise_dbfs)) else round(float(noise_dbfs), 2),
                "speech_rms": None if np.isnan(speech_rms) else round(float(speech_rms), 6),
                "snr_db": snr_db,

                # VAD params
                "vad_mode": vad_mode,
                "frame_ms": frame_ms,
            }
            if wer_val is not None:
                out["WER"] = round(float(wer_val), 4)

            out_rows.append(out)

        except Exception as e:
            print(f"[ERR] {audio_path}: {e}")

    return pd.DataFrame(out_rows)

# ============== run ==============
if __name__ == "__main__":
   
    csv_file = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1high_wer_speakersnewaudio.csv"
    
    audio_folder = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull"

    df_results = analyze_audio(csv_file, audio_folder)  
    out_csv = "/home/jacobala@alabsad.fau.de/AOT/1newvad_wer_analysisfornew_audio.csv"
    if not df_results.empty:
        df_results.to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")
    else:
        print("[WARN] No results to save.")
