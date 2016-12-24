# MIR3
## Music Instrument Recognition using convolutional neural networks

The repository contains the source code for music instrument recognition implemented in python/tensorflow v0.12.
The model was trained using the MedleyDB dataset

## Prediction
- After cloning the repository, run sh_download_model_1.sh
- Then run **predict.py <path-to-wave-file>**

## Training
- Step 1. run **data_chop_medley_db.py** to chop long tracks into normalized 3-seconds chunks, excluding silence
- Step 2. run **data_prepare_mini_experiment.py** to prepare data subsets to be used for training and validation
- Step 3. run **data_prepare_spectrograms.py** to do resampling, FFT and MFCC 
- Step 4. run **train.py** for training

## Experiments
run **MFCC.ipynb** and **MedleyExamples.ipynb**

