import numpy as np
from sklearn.model_selection import train_test_split
import glob
from pathlib import Path

# Load all preprocessed data
output_dir = Path("preprocessed_data")
data_files = glob.glob(str(output_dir / "*_data.npy"))
all_data = []
all_labels = []

for f in data_files:
    data = np.load(f)
    labels = np.load(f.replace("_data.npy", "_labels.npy"))
    all_data.append(data)
    all_labels.append(labels)

data = np.concatenate(all_data, axis=0)
labels = np.concatenate(all_labels, axis=0)

# Split into train, val, test
X_temp, X_test, y_temp, y_test = train_test_split(
    data, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42  # 0.15/0.85 ≈ 0.1765
)

# Save splits
np.save("train_data.npy", X_train)
np.save("train_labels.npy", y_train)
np.save("val_data.npy", X_val)
np.save("val_labels.npy", y_val)
np.save("test_data.npy", X_test)
np.save("test_labels.npy", y_test)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")