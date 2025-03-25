import mne
import numpy as np
from pathlib import Path

# Define paths
data_dir = Path("chbmit")
output_base_dir = Path("preprocessed_data")
output_base_dir.mkdir(exist_ok=True)

# Parameters
window_size = 256  # 1 second at 256 Hz
stride = 128       # 50% overlap
chunk_duration = 600  # 10 minutes in seconds (adjustable)
sfreq = 256  # Sampling frequency (fixed for CHB-MIT)

# Process each case
for case_dir in data_dir.glob("chb*"):
    case = case_dir.name
    output_dir = output_base_dir / case
    output_dir.mkdir(exist_ok=True)
    edf_files = sorted(case_dir.glob("*.edf"))
    seizure_file = case_dir / f"{case}-summary.txt"  # Annotations

    # Parse seizure times from summary file
    seizure_times = {}
    with open(seizure_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "File Name:" in line:
                file_name = line.split("File Name: ")[1].strip()
                if i + 1 < len(lines) and "Seizure Start Time" in lines[i + 1]:
                    start = int(lines[i + 1].split(": ")[1].split()[0])
                    end = int(lines[i + 2].split(": ")[1].split()[0])
                    seizure_times[file_name] = (start, end)

    # Process each EDF file
    for edf_path in edf_files:
        print(f"Processing {edf_path.name}")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)  # Load lazily
        n_samples = int(raw.n_times)
        chunk_samples = chunk_duration * sfreq  # Samples per chunk

        # Process in chunks
        for chunk_start in range(0, n_samples, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, n_samples)
            # Load chunk data
            data = raw.get_data(picks="eeg", start=chunk_start, stop=chunk_end)[:23]
            chunk_time_start = chunk_start / sfreq

            # Windowing
            windows = []
            labels = []
            for start in range(0, data.shape[1] - window_size + 1, stride):
                window = data[:, start:start + window_size]
                # Normalize window
                window_mean = window.mean()
                window_std = window.std()
                if window_std == 0:  # Avoid division by zero
                    window_std = 1e-10
                window = (window - window_mean) / window_std

                # Label based on seizure times
                window_time = chunk_time_start + (start / sfreq)
                is_seizure = False
                if edf_path.name in seizure_times:
                    s_start, s_end = seizure_times[edf_path.name]
                    if s_start <= window_time <= s_end:
                        is_seizure = True
                windows.append(window)
                labels.append(1 if is_seizure else 0)

            # Save chunk if windows exist
            if windows:
                chunk_idx = chunk_start // chunk_samples
                np.save(output_dir / f"{edf_path.stem}_chunk{chunk_idx}_data.npy", np.array(windows))
                np.save(output_dir / f"{edf_path.stem}_chunk{chunk_idx}_labels.npy", np.array(labels))
                print(f"Saved chunk {chunk_idx} for {edf_path.name}")

        raw.close()  # Free memory
    print(f"Completed case {case}")