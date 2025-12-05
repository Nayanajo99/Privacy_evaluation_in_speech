import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import ast
import shutil


import sys

# === CONFIGURATION ===
xlsx_path = "/home/jacobala@alabsad.fau.de/AOT/testnew.xlsx"
audio_base_dir = "/home/jacobala@alabsad.fau.de/AOT/SLUE_TEST"
output_dir = "output_audiofilesnewfull"
sample_rate = 16000
target_duration_sec = 30
min_clips = 390

# === Validate input paths ===
if not os.path.isfile(xlsx_path):
    sys.exit(f" Excel file not found: {xlsx_path}")

if not os.path.isdir(audio_base_dir):
    sys.exit(f" Audio base directory not found: {audio_base_dir}")

print(f" Excel file: {xlsx_path}")
print(f" Audio folder: {audio_base_dir}")
print(f" Output folder will be: {output_dir}")

def set_random_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def safe_parse_ner(value):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    return []


def shift_ner_offsets(ner_list, char_offset):
    shifted = []
    for item in ner_list:
        if isinstance(item, list) and len(item) == 3:
            label, start_char, length = item
            shifted.append([label, start_char + char_offset, length])
    return shifted


def mix_audio_from_excel(xlsx_path, audio_base_dir, output_dir, sample_rate=16000, target_duration_sec=30, min_clips=390):
    set_random_seed(42)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    df = pd.read_excel(xlsx_path, sheet_name=0, engine='openpyxl')
    df = df.iloc[1:].copy()
    df.columns = ['speaker_id_group', 'id', 'raw_text', 'normalized_text', 'speaker_id', 'split', 'raw_ner', 'normalized_ner']
    df = df[['speaker_id', 'id', 'raw_text', 'normalized_text', 'raw_ner', 'normalized_ner']]
    df = df.dropna(subset=['id', 'raw_text'])

    df['audio'] = df['id'].astype(str).apply(lambda x: os.path.basename(x).strip())
    df['audio'] = df['audio'].apply(lambda x: x if x.endswith('.ogg') else f"{x}.ogg")

    df['has_ner'] = df.apply(lambda row: bool(safe_parse_ner(row['raw_ner'])) or bool(safe_parse_ner(row['normalized_ner'])), axis=1)
    valid_speaker_ids = df[df['has_ner']]['speaker_id'].unique()
    df = df[df['speaker_id'].isin(valid_speaker_ids)]

    all_groups = list(df.groupby("speaker_id"))
    random.shuffle(all_groups)

    csv_rows = []
    clip_counter = 0
    round_counter = 0

    while clip_counter < min_clips and round_counter < 20:
        print(f"\nPass {round_counter + 1}...")

        for speaker_id, group in all_groups:
            group = group.sample(frac=1, random_state=round_counter).reset_index(drop=True)
            ptr = 0

            while ptr < len(group) and clip_counter < min_clips:
                combined_audio = []
                combined_raw_text = ""
                combined_norm_text = ""
                shifted_raw_ner = []
                shifted_norm_ner = []
                total_audio_len = 0
                char_offset_raw = 0
                char_offset_norm = 0
                ner_found = False
                used_ids = []

                while ptr < len(group) and total_audio_len < target_duration_sec * sample_rate:
                    row = group.iloc[ptr]
                    ptr += 1

                    audio_file = str(row['audio']).replace('\\', '/').lstrip('a/').lstrip('/')
                    audio_path = os.path.join(audio_base_dir, audio_file)

                    raw_text = str(row['raw_text']).strip()
                    norm_text = str(row['normalized_text']).strip()
                    raw_ner_tokens = safe_parse_ner(row['raw_ner'])
                    norm_ner_tokens = safe_parse_ner(row['normalized_ner'])

                    if len(raw_text.split()) < 2:
                        continue

                    try:
                        audio, sr = librosa.load(audio_path, sr=sample_rate)
                    except Exception as e:
                        print(f"Could not load {audio_path}: {e}")
                        continue

                    audio_len = len(audio)
                    if total_audio_len + audio_len > target_duration_sec * sample_rate:
                        remaining = target_duration_sec * sample_rate - total_audio_len
                        if remaining <= 0:
                            break
                        ratio = remaining / audio_len
                        word_count = max(1, int(len(raw_text.split()) * ratio))
                        raw_text = " ".join(raw_text.split()[:word_count])
                        norm_text = " ".join(norm_text.split()[:word_count])
                        audio = audio[:remaining]
                        audio_len = len(audio)
                        raw_ner_tokens = [ent for ent in raw_ner_tokens if ent[1] < len(raw_text)]
                        norm_ner_tokens = [ent for ent in norm_ner_tokens if ent[1] < len(norm_text)]

                    if not raw_ner_tokens and not norm_ner_tokens:
                        continue

                    shifted_raw_ner.extend(shift_ner_offsets(raw_ner_tokens, char_offset_raw))
                    shifted_norm_ner.extend(shift_ner_offsets(norm_ner_tokens, char_offset_norm))

                    combined_raw_text += raw_text + " "
                    combined_norm_text += norm_text + " "
                    combined_audio.append(audio)
                    char_offset_raw += len(raw_text) + 1
                    char_offset_norm += len(norm_text) + 1
                    total_audio_len += audio_len
                    ner_found = True
                    used_ids.append(row['id'])

                if not ner_found or total_audio_len < target_duration_sec * sample_rate:
                    continue

                audio_concat = np.concatenate(combined_audio)
                base_filename = f"clip_{clip_counter:04d}_speaker_{int(float(speaker_id))}"
                audio_out = os.path.join(output_dir, f"{base_filename}.wav")
                txt_out = os.path.join(output_dir, f"{base_filename}.txt")

                sf.write(audio_out, audio_concat, sample_rate)
                with open(txt_out, "w") as f:
                    f.write(combined_raw_text.strip() + "\n")

                csv_rows.append({
                    'speaker_id': speaker_id,
                    'audio_filename': f"{base_filename}.wav",
                    'combined_from_ids': ", ".join(used_ids),
                    'raw_text': combined_raw_text.strip(),
                    'normalized_text': combined_norm_text.strip(),
                    'raw_ner': str(shifted_raw_ner),
                    'normalized_ner': str(shifted_norm_ner)
                })

                print(f"Saved: {audio_out}")
                clip_counter += 1

                if clip_counter >= min_clips:
                    print(f"\nReached {min_clips} clips — stopping.")
                    break

            if clip_counter >= min_clips:
                break

        round_counter += 1

    csv_output_path = os.path.join(output_dir, "all_transcriptsnw.csv")
    pd.DataFrame(csv_rows).to_csv(csv_output_path, index=False, encoding='utf-8')
    print(f"\nFinal CSV saved to: {csv_output_path}")
    print(f"Done. Created exactly {clip_counter} valid 30s audio clips with NER.")
    return csv_output_path

mix_audio_from_excel(
   xlsx_path=xlsx_path,
   audio_base_dir=audio_base_dir,
   output_dir=output_dir,
   sample_rate=sample_rate,
   target_duration_sec=target_duration_sec,
   min_clips=min_clips
)