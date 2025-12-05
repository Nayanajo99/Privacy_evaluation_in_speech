import os
import numpy as np
import soundfile as sf
import shutil
np.random.seed(42)

def sample_audio_time_domain(y, sr, keep_ratio, frame_length_ms):
    frame_length = int((frame_length_ms / 1000) * sr)
    n_frames = len(y) // frame_length
    remainder = len(y) % frame_length

    if remainder != 0:
        pad_length = frame_length - remainder
        y_padded = np.pad(y, (0, pad_length), mode='constant')
    else:
        y_padded = y

    frames = y_padded.reshape(-1, frame_length)
    n_keep = int(frames.shape[0] * keep_ratio)

    if frames.shape[0] == 0 or n_keep == 0:
        return np.array([], dtype=np.float32), sr

    sampled_indices = np.sort(np.random.choice(frames.shape[0], n_keep, replace=False))
    sampled_frames = frames[sampled_indices]
    y_concatenated = sampled_frames.flatten()

    print(f"Original length: {len(y)}, Sampled length: {len(y_concatenated)} for keep_ratio={keep_ratio}")
    return y_concatenated, sr


def process_audio_files_time_domain_multi_snr(base_input_dir, subsample_rates, frame_length_ms):
    base_results_dir = os.path.join(os.getcwd(), '1NewmixsampleResultsSNR15')
    os.makedirs(base_results_dir, exist_ok=True)
    print(f"Base results will be saved in: {base_results_dir}")

    snr_dirs = [d for d in os.listdir(base_input_dir) if d.startswith("SNR_") and os.path.isdir(os.path.join(base_input_dir, d))]

    for snr_dir in snr_dirs:
        snr_path = os.path.join(base_input_dir, snr_dir)
        print(f"\nProcessing SNR folder: {snr_dir}")

        for rate in subsample_rates:
            keep_ratio = 1.0 - (rate / 100.0)
            rate_dir = os.path.join(base_results_dir, snr_dir, f"rate_{rate}")
            os.makedirs(rate_dir, exist_ok=True)
            print(f"Sampling rate {rate}% -> saving in {rate_dir}")

            if rate == 0:
                print(" Sampling rate 0% = baseline (copying original files without processing)")
                for file in os.listdir(snr_path):
                    if file.endswith('.wav') or file.endswith('.txt') or file.endswith('.log'):
                        src = os.path.join(snr_path, file)
                        dst = os.path.join(rate_dir, file)
                        shutil.copy2(src, dst)
                        print(f"Copied {file} -> {dst}")
                continue  

            for file in os.listdir(snr_path):
                if file.endswith('.wav'):
                    base_name = os.path.splitext(file)[0]
                    prefix = base_name.split('_')[0]
                    audio_path = os.path.join(snr_path, file)
                    transcript_path = os.path.join(snr_path, f"{base_name}.txt")
                    log_path = os.path.join(snr_path, f"{base_name}_log.txt")

                    y, sr = sf.read(audio_path)
                    y_sampled, sr = sample_audio_time_domain(y, sr, keep_ratio, frame_length_ms)

                    audio_output_path = os.path.join(rate_dir, f"{base_name}.wav")
                    transcript_output_path = os.path.join(rate_dir, f"{base_name}.txt")
                    log_output_path = os.path.join(rate_dir, f"{base_name}_log.txt")

                    sf.write(audio_output_path, y_sampled, sr)
                    print(f"Processed {file} -> {audio_output_path}")

                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r') as transcript_file:
                            transcript = transcript_file.read()
                        with open(transcript_output_path, 'w') as transcript_out:
                            transcript_out.write(transcript)

                    if os.path.exists(log_path):
                        with open(log_path, 'r') as log_file:
                            log = log_file.read()
                        with open(log_output_path, 'w') as log_out:
                            log_out.write(log)

    return base_results_dir


# === CONFIGURATION ===
root_directory = "/home/jacobala@alabsad.fau.de/AOT/Obfuscation_Techniques/1newoutput_combined_multi_snr"
subsample_rates = [0, 10, 20, 30, 40, 50]
frame_length_ms = 128

# === RUN ===
output_path = process_audio_files_time_domain_multi_snr(root_directory, subsample_rates, frame_length_ms)
print(f"\nTime-domain sampling complete. Results saved in: {output_path}")
