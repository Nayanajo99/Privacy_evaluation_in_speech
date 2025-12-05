import os
import random
import numpy as np
import librosa
import soundfile as sf

random.seed(42)

# ----------------------------- LOADERS -----------------------------

def load_audio_and_transcripts(audio_dir, sampling_rate, limit=390):
    """
    Load up to `limit` speech wav files and their raw-line transcripts.
    Returns: list of (audio_np, transcript_lines, audio_filename, transcript_filename)
    """
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    random.shuffle(audio_files)
    audio_files = audio_files[:limit]

    transcript_files = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(audio_dir) if f.endswith('.txt')
    }

    pairs = []
    for audio_file in audio_files:
        base = os.path.splitext(audio_file)[0]
        transcript_file = transcript_files.get(base)

        # load speech (background) at native sr, resample if needed
        audio_path = os.path.join(audio_dir, audio_file)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        if sr != sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)

        transcript = []
        if transcript_file:
            transcript_path = os.path.join(audio_dir, transcript_file)
            with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
                transcript = [line.strip() for line in f]

        pairs.append((audio, transcript, audio_file, transcript_file))
    return pairs

# ----------------------------- MIXING -----------------------------

def mix_audio(new_audio, mixed_audio, snr_level_db):
    """
    Mix ASC foreground (new_audio) with speech background (mixed_audio) at target SNR (dB).
    SNR definition: SNR = P_foreground / P_background_after_scaling.
    Keep ASC level fixed and scale speech to meet the target SNR.

    Returns: mixed_signal (np.ndarray)
    """
    # Time-align to same length
    N = min(len(new_audio), len(mixed_audio))
    fg = new_audio[:N]           # ASC (fixed)
    bg = mixed_audio[:N]         # speech (to be scaled)

    # RMS
    rms_fg = np.sqrt(np.mean(fg**2)) if N > 0 else 0.0
    rms_bg = np.sqrt(np.mean(bg**2)) if N > 0 else 0.0

    # Edge case: both silent
    if rms_fg == 0 and rms_bg == 0:
        return np.zeros_like(fg)

   
    #   rms_bg_req = rms_fg / (10^(SNR/20))
    rms_bg_req = rms_fg / (10.0 ** (snr_level_db / 20.0))
    a = 0.0 if rms_bg == 0 else (rms_bg_req / rms_bg)

    # Mix (ASC fixed, speech scaled)
    y = fg + a * bg

    
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1.0:
        y = y / peak

    return y

# ----------------------------- SAVE -----------------------------

def save_pair(snr_output_dir, y, transcript_lines, audio_filename, transcript_filename, sr):
    os.makedirs(snr_output_dir, exist_ok=True)
    sf.write(os.path.join(snr_output_dir, audio_filename), y, sr)
    with open(os.path.join(snr_output_dir, transcript_filename), 'w', encoding='utf-8') as f:
        for line in transcript_lines:
            f.write(f"{line}\n")

# ----------------------------- PIPELINE -----------------------------

def process_and_mix_audios(new_audio_dir, mixed_audio_dir, output_dir,
                           sampling_rate=16000, snr_levels=(0,5,10,15,20), max_files=390):
    """
    - new_audio_dir: ASC foreground (.wav)
    - mixed_audio_dir: speech background (.wav) + transcripts (.txt as raw lines)
    - output_dir: root; per-SNR subfolders SNR_<L> are created
    - snr_levels: target SNR values (dB)
    - max_files: limit number of speech files loaded/paired
    """
    # Load speech+transcripts
    speech_data = load_audio_and_transcripts(mixed_audio_dir, sampling_rate, limit=max_files)
    if not speech_data:
        raise RuntimeError(f"No speech files found in: {mixed_audio_dir}")

    # Load ASC list
    asc_files = sorted([f for f in os.listdir(new_audio_dir) if f.endswith('.wav')])
    if not asc_files:
        raise RuntimeError(f"No ASC files found in: {new_audio_dir}")

    # Pair count
    pair_count = min(len(asc_files), len(speech_data))
    asc_files = asc_files[:pair_count]

    print(f"Pairing {pair_count} ASC files with {pair_count} speech files, across SNRs {snr_levels}.")

    # Prepare SNR folders
    snr_dirs = {L: os.path.join(output_dir, f"SNR_{L}") for L in snr_levels}
    for d in snr_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Shuffle speech then pair one-to-one with ASC
    random.shuffle(speech_data)
    pairs = list(zip(asc_files, speech_data))  # (asc_file, (speech_audio, transcript_lines, speech_fn, transcript_fn))

    for asc_file, (speech_audio, transcript_lines, speech_fn, speech_tf) in pairs:
        # Load ASC (foreground)
        asc_path = os.path.join(new_audio_dir, asc_file)
        asc_audio, sr = librosa.load(asc_path, sr=None, mono=True)
        if sr != sampling_rate:
            asc_audio = librosa.resample(asc_audio, orig_sr=sr, target_sr=sampling_rate)

        base_audio_name = os.path.splitext(speech_fn)[0]  # speech
        base_asc_name   = os.path.splitext(asc_file)[0]   # ASC

        for L in snr_levels:
            y = mix_audio(asc_audio, speech_audio, L)

            out_wav = f"{base_audio_name}_with_{base_asc_name}_snr{L}.wav"
            out_txt = f"{base_audio_name}_with_{base_asc_name}_snr{L}.txt"

            save_pair(snr_dirs[L], y, transcript_lines, out_wav, out_txt, sampling_rate)

    print(f"Done. Outputs under: {output_dir}")
    for L in snr_levels:
        print(f"  - {snr_dirs[L]}")




new_audio_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/TUTASC"  # ASC foreground
mixed_audio_dir = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/output_audiofilesnewfull"         # speech background 
output_dir = "newoutput_combined_multi_snr"
sampling_rate = 32000
snr_levels = (0, 5, 10, 15, 20)
max_files = 390  # max speech files 

process_and_mix_audios(new_audio_dir, mixed_audio_dir, output_dir,
                       sampling_rate=sampling_rate,
                       snr_levels=snr_levels,
                       max_files=max_files)
