import mne
import numpy as np
from pathlib import Path

# Define paths
data_dir = Path("chbmit")
output_dir = Path("preprocessed_data")
output_dir.mkdir(exist_ok=True)

# Parameters
window_size = 256  # 1 second at 256 Hz
stride = 128       # 50% overlap

# Process each case
for case_dir in data_dir.glob("chb*"):
    case = case_dir.name
    edf_files = list(case_dir.glob("*.edf"))
    seizure_file = case_dir / f"{case}-summary.txt"  # Annotations

    # Parse seizure times from summary file
    seizure_times = {}
    with open(seizure_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "File Name:" in line:
                file_name = line.split("File Name: ")[1].strip()
                if "Seizure Start Time" in lines[i + 1]:
                    start = int(lines[i + 1].split(": ")[1].split()[0])
                    end = int(lines[i + 2].split(": ")[1].split()[0])
                    seizure_times[file_name] = (start, end)

    # Process each EDF file
    for edf_path in edf_files:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data = raw.get_data(picks="eeg")[:23]  # First 23 EEG channels
        sfreq = raw.info["sfreq"]
        assert sfreq == 256, f"Unexpected sampling rate in {edf_path}"

        # Windowing
        n_samples = data.shape[1]
        windows = []
        labels = []
        for start in range(0, n_samples - window_size + 1, stride):
            window = data[:, start:start + window_size]
            # Normalize window
            window = (window - window.mean()) / window.std()
            # Label based on seizure times
            window_time = start / sfreq
            is_seizure = False
            if edf_path.name in seizure_times:
                s_start, s_end = seizure_times[edf_path.name]
                if s_start <= window_time <= s_end:
                    is_seizure = True
            windows.append(window)
            labels.append(1 if is_seizure else 0)

        # Save preprocessed data
        np.save(output_dir / f"{case}_{edf_path.stem}_data.npy", np.array(windows))
        np.save(output_dir / f"{case}_{edf_path.stem}_labels.npy", np.array(labels))
        print(f"Processed {edf_path.name}")