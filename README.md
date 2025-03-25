# Seizure Detection Project
This project implements seizure detection using a Lightweight 1D CNN on the CHB-MIT EEG dataset. Up to Phase 2, it covers dataset preprocessing, model training, and quantization.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download CHB-MIT dataset manually or via wfdb.
3. Run scripts in order: `preprocess.py`, `split_dataset.py`, `train_model.py`, `evaluate_quantize.py`.

## Next Steps
Phase 3: FPGA implementation on PYNQ board.