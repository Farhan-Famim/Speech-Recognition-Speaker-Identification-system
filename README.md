# Speech Recognition and Speaker Classification System

A machine learning course project for **CSE445 – Machine Learning**. The project combines a pretrained **Whisper** automatic speech recognition (ASR) model with an **MFCC-based Support Vector Machine (SVM)** classifier to transcribe audio and identify which of the known speakers produced a single-speaker recording.

> **Project status:** Working prototype. The current implementation has been tested with a small dataset of 3 speakers and 5 recordings per speaker. It is intended to be expanded to a larger dataset (eventually 10 speakers or more).

## Table of Contents

- [Overview](#overview)
- [What the System Does](#what-the-system-does)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Feature Caching and Dataset Updates](#feature-caching-and-dataset-updates)
- [Evaluation](#evaluation)
- [Current Limitations](#current-limitations)
- [Possible Future Improvements](#possible-future-improvements)
- [Reproducibility and Repository Notes](#reproducibility-and-repository-notes)
- [References](#references)

## Overview

The project has two main machine learning components:

1. **Speech recognition:** Whisper converts an input audio recording into text.
2. **Speaker classification:** An SVM model classifies the recording as belonging to one of the speakers represented in the training dataset.

The two components are combined to produce an output containing both the predicted speaker label and the recognized text.

The project uses transfer learning in the broad practical sense of using a pretrained ASR model rather than building a speech recognition model from scratch. **The current implementation uses Whisper's pretrained `base` model; Whisper fine-tuning has not yet been implemented.** The speaker classifier, on the other hand, is trained locally using the project's collected audio samples.

## What the System Does

Given an audio file containing one speaker, the system can:

- Transcribe the spoken content into text using Whisper.
- Extract MFCC (Mel-Frequency Cepstral Coefficient) features from the audio.
- Use an SVM classifier to predict the identity label of the speaker, provided that the speaker was included in the training dataset.
- Compare Whisper's transcription with a manually prepared reference transcript using Word Error Rate (WER).
- Cache extracted MFCC features in NumPy files so that they do not need to be recalculated every time the training script runs.

### Important distinction: speaker classification vs. speaker diarization

The current SVM component performs **closed-set speaker classification**. It assigns one speaker label to an audio file based on the voices it learned during training.

It does **not** automatically find speaker changes inside a recording, split a conversation into speaker turns, or handle overlapping speech. For example, if a recording contains Speaker A followed by Speaker B, the current SVM pipeline does not segment the recording and label each part separately. That task is called **speaker diarization** and would require an additional segmentation/diarization component, such as `pyannote.audio`, or a separate segmentation strategy.

The current workflow therefore assumes that each audio file used for speaker classification contains speech from **one speaker only**.

## How It Works

### 1. Speech-to-text with Whisper

- Load the pretrained Whisper `base` model once.
- Pass an audio file to `model.transcribe()`.
- Read the recognized text from the returned result.

Whisper is used for transcription; it is not currently fine-tuned on the collected speaker dataset.

### 2. MFCC feature extraction

The SVM classifier cannot learn directly from the audio file path or raw waveform in the current implementation. Each recording is converted into a numerical feature vector:

1. Load the audio using `librosa` at a sample rate of 16 kHz.
2. Extract 13 MFCC coefficients for the audio frames.
3. Calculate the mean of each coefficient across time.
4. Use the resulting 13-value vector as the representation of that recording.

This produces a fixed-size feature vector, regardless of the recording duration. The mean-MFCC representation is intentionally simple and lightweight, but it also discards some information about how voice characteristics change over time.

### 3. SVM speaker classification

- Each audio file's MFCC vector becomes one row in the feature matrix `X`.
- The speaker folder name becomes that sample's label in `y` (for example, `speaker_1`).
- An SVM with a linear kernel (`SVC(kernel="linear")`) is trained on the extracted features.
- The trained classifier predicts one of the speaker labels learned during training.

### 4. Transcription evaluation

The ASR output is compared with the corresponding reference transcript using **Word Error Rate (WER)**:

\[
WER = \frac{S + D + I}{N}
\]

Where:

- `S` = substitutions (a reference word is replaced by another word)
- `D` = deletions (a reference word is missing from the prediction)
- `I` = insertions (an extra word appears in the prediction)
- `N` = number of words in the reference transcript

Lower WER indicates better transcription performance. The evaluation code normalizes text by converting it to lowercase and removing punctuation/extra whitespace before calculating WER. This avoids penalizing differences that are only due to capitalization or punctuation. It does **not** automatically equate every formatting difference, such as `40,000` and `forty thousand`.

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| OpenAI Whisper (`base`) | Automatic speech recognition (audio-to-text) |
| PyTorch | Backend used by Whisper |
| librosa | Audio loading and MFCC feature extraction |
| NumPy | Feature arrays and cached feature data |
| scikit-learn | SVM training, prediction, and accuracy evaluation |
| jiwer | Word Error Rate calculation |
| joblib | Optional model serialization if the trained SVM is saved for later use |
| FFmpeg | Audio decoding support required by Whisper |

## Project Structure

A suggested repository structure is shown below. Keep the actual filenames consistent with the scripts in the repository.

```text
speech-recognition-project/
│
├── dataset/
│   ├── speaker_1/
│   │   ├── audio_1.wav
│   │   ├── audio_2.wav
│   │   └── ...
│   ├── speaker_2/
│   │   ├── audio_1.wav
│   │   └── ...
│   ├── speaker_3/
│   │   ├── audio_1.wav
│   │   └── ...
│   └── transcripts.txt
│
├── feature_extraction.py
├── train_svm.py
├── final_system.py
├── features_X.npy          # generated feature cache
├── labels_y.npy            # generated speaker-label cache
├── svm_speaker_model.pkl   # generated only if model-saving is enabled
├── requirements.txt        # recommended dependency list
└── README.md
```

Generated cache/model files do not have to be committed to GitHub. They can be regenerated locally from the dataset and scripts.

## Dataset Preparation

### Folder organization

Each speaker should have a separate folder. The folder name is used as the speaker's class label by `train_svm.py`.

For example:

```text
dataset/
├── speaker_1/
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── audio_3.wav
├── speaker_2/
│   ├── audio_1.wav
│   ├── audio_2.wav
│   └── audio_3.wav
└── transcripts.txt
```

### Recording recommendations

For consistent experiments, try to keep recordings:

- In WAV format.
- Mono (one channel), where possible.
- Recorded in a reasonably quiet environment.
- Clear and free from clipping or excessive background noise.
- Long enough to contain useful speech; very short clips may not provide reliable speaker characteristics.
- Recorded at similar conditions where practical, while still including natural variation in speaking style.

The feature extraction script loads audio at 16 kHz, so recordings with other sample rates are resampled by `librosa` during loading.

### Transcript file format

The `transcripts.txt` file contains one record per audio file, using this format:

```text
relative/path/to/audio.wav|The exact spoken sentence goes here.
```

Example:

```text
speaker_1/audio_1.wav|The story of writing, astronomy, and law.
speaker_1/audio_2.wav|Mesopotamia is an exceedingly fertile plain situated between the Tigris and Euphrates Rivers.
speaker_2/audio_1.wav|If we are going to study world history, then let's start our journey at the beginning.
```

Guidelines:

- The path before `|` must match the audio file's location relative to the `dataset/` directory.
- Use one line per audio file.
- Keep the transcript as close as possible to the words actually spoken.
- Avoid adding explanatory text or speaker labels after the separator.
- Save the file as UTF-8.
- The current speaker-training script uses the speaker folder names for labels; it does not use `transcripts.txt` to train the SVM. The transcript file is used by the Whisper evaluation script.

### Dataset split for a proper experiment

For a stronger evaluation, collect separate recordings for training and testing. Ideally, the test recordings should be new recordings from the **same speaker identities** that appear in training, but should not be used to train the classifier.

If the test set contains entirely new people who were not represented in training, this SVM cannot identify them by their real identity. It only predicts among the speaker classes it learned. A separate unknown-speaker/rejection mechanism or a different experimental setup would be needed to handle unseen identities.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd speech-recognition-project
```

### 2. Create and activate a virtual environment (recommended)

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Python dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the packages used by the project:

```bash
pip install openai-whisper torch librosa scikit-learn numpy jiwer joblib
```

Whisper also requires **FFmpeg** to be installed and available on the system PATH. On Windows, install an FFmpeg build and add its `bin` directory to PATH. Then open a new terminal and verify that this command works:

```bash
ffmpeg -version
```

The first Whisper run may download the selected model weights. An internet connection is required for that initial download unless the model is already present in the local cache.

> Dependency compatibility can vary by operating system and package release. Use a virtual environment so that project dependencies do not interfere with other Python projects.

## Running the Project

Run commands from the repository's root directory so that paths such as `dataset/` resolve correctly.

### 1. Test MFCC feature extraction

Run:

```bash
python feature_extraction.py
```

The test code in `feature_extraction.py` should load a sample audio file and print a feature vector with 13 values. Its shape should be:

```text
(13,)
```

### 2. Train and evaluate the SVM classifier

Run:

```bash
python train_svm.py
```

The script:

1. Checks whether `features_X.npy` and `labels_y.npy` already exist.
2. If they do not exist, scans the speaker folders, extracts MFCC features, and saves the feature matrix and labels.
3. If the cache files exist, loads them instead of extracting MFCC features again.
4. Splits the samples into training and testing subsets using the configured `train_test_split` settings.
5. Trains a linear-kernel SVM.
6. Reports classification accuracy.
7. Runs a speaker prediction on the test audio path configured in the script.

With the initial dataset of 3 speakers and 5 recordings per speaker, the feature matrix should have 15 rows and 13 columns:

```text
X shape: (15, 13)
y shape: (15,)
```

The exact accuracy may vary if the dataset, split, or model settings change.

### 3. Transcribe an audio file with Whisper

The ASR script loads Whisper's `base` model and calls `model.transcribe(audio_path)`. Set the input path to the audio file you want to transcribe, then run the script:

```bash
python <YOUR_WHISPER_SCRIPT>.py
```

Replace `<YOUR_WHISPER_SCRIPT>.py` with the actual filename used in your repository.

### 4. Run the integrated system

The integrated script is intended to:

1. Receive one audio file.
2. Transcribe it using Whisper.
3. Extract its MFCC features.
4. Predict its speaker using the trained SVM.
5. Print the speaker label and recognized text.

Example output:

```text
Processing: dataset/speaker_2/audio_2.wav

Final Output:
speaker_2: If we are going to study world history, then let's start our journey...
```

If `final_system.py` loads `svm_speaker_model.pkl`, the training script must first save the trained classifier to that file (for example, using `joblib.dump(model, "svm_speaker_model.pkl")` after `model.fit(...)`). **Saving the MFCC cache is not the same as saving the trained SVM model.** The cache stores `X` and `y`; the classifier must still be trained from those features, unless the fitted model itself is also serialized. After saving the model, run:

```bash
python final_system.py
```

## Feature Caching and Dataset Updates

The project caches extracted features in two files:

- `features_X.npy`: MFCC feature matrix.
- `labels_y.npy`: corresponding speaker labels.

On the first run, the training script extracts features from the WAV files and saves them. On later runs, it loads those cached arrays, avoiding repeated MFCC extraction.

### What happens when new recordings or speakers are added?

The current cache implementation checks whether the two cache files exist. It does **not** automatically check whether the dataset has changed. Therefore, newly added recordings or speaker folders will not be included while the old cache is being loaded.

To rebuild the cache after changing the dataset, delete both cache files and rerun `train_svm.py`:

On Windows Command Prompt:

```bat
del features_X.npy
del labels_y.npy
python train_svm.py
```

On PowerShell:

```powershell
Remove-Item features_X.npy, labels_y.npy
python train_svm.py
```

The script will then extract features from all available speaker folders and train a new classifier using the refreshed dataset.

## Evaluation

The project uses two different evaluation measures because it contains two different tasks.

### Speech recognition: Word Error Rate (WER)

WER measures the difference between the reference transcript and the text predicted by Whisper. A WER of `0.00` represents an exact word-level match after the configured text normalization. Lower is better.

The normalization currently used in the evaluation code:

- Converts text to lowercase.
- Removes punctuation.
- Removes repeated/leading/trailing whitespace.

Punctuation and capitalization differences therefore do not affect the score. Word substitutions, missing words, and additional words still affect it. Number formatting may also affect WER unless a separate number-normalization step is implemented.

### Speaker classification: Accuracy

The SVM script reports classification accuracy on the held-out portion created by `train_test_split`. Accuracy is the fraction of test samples assigned the correct speaker label.

An accuracy of `1.0` means all samples in that particular test split were classified correctly. It does **not** guarantee that the model will achieve 100% accuracy on new recordings. The initial experiment contains only 15 recordings, so its evaluation is preliminary and may be highly sensitive to which samples fall into the test split.

For a more reliable result, expand the number and diversity of recordings per speaker, use a stratified split where appropriate, and evaluate on a separate set of recordings that was not used for feature extraction/model fitting decisions.

## Current Limitations

- **Single-speaker input assumption:** the current SVM predicts one speaker for an entire audio file. It does not diarize a recording containing multiple speakers.
- **Known-speaker classification only:** the SVM can classify speakers represented during training. It is not designed to identify a completely unseen person.
- **Small initial dataset:** the current prototype uses 3 speakers and 5 audio samples per speaker. This is useful for development but insufficient for strong generalization claims.
- **Simple speaker representation:** the current classifier uses the mean of 13 MFCC coefficients. It does not use specialized speaker embeddings or a deep speaker-recognition network.
- **Whisper is not fine-tuned:** the current speech recognition component uses the pretrained Whisper `base` model directly.
- **Feature cache refresh is manual:** after adding or replacing training recordings, the cached `.npy` files must be deleted before rebuilding the dataset.
- **Recording conditions matter:** background noise, microphone quality, speaking distance, accent, pace, and room acoustics can affect both transcription and speaker classification.

## Possible Future Improvements

- Expand the dataset to 10 or more speakers and collect multiple recordings per speaker across different sentences and recording sessions.
- Create a separate test set containing new recordings from the same enrolled speakers.
- Implement automatic cache invalidation or append only features for new audio files.
- Save the trained SVM model with `joblib` so inference does not require retraining each time.
- Experiment with MFCC normalization, additional acoustic features, different SVM kernels, and hyperparameter tuning.
- Compare speaker classification performance with more advanced speaker embeddings.
- Add a speaker diarization component to detect speaker changes and label segments in multi-speaker recordings.
- Investigate Whisper fine-tuning or domain adaptation if the dataset and compute resources justify it.
- Add a simple graphical interface or microphone input for interactive testing.

## Reproducibility and Repository Notes

Before publishing the repository:

- Include the source files and a clear dependency list.
- Do not commit your virtual environment (`venv/`).
- Consider excluding generated cache/model files (`features_X.npy`, `labels_y.npy`, `svm_speaker_model.pkl`) if they can be recreated locally.
- Include a small sample dataset only when you have permission to distribute the recordings and transcripts.
- Do not upload other people's voice recordings without their informed consent. Check the license or usage rights for any audio or transcript taken from external sources.
- If the full dataset is too large or cannot be redistributed, explain how to arrange the dataset locally and provide a small permitted sample or a data-collection guide instead.

A suggested `.gitignore` is:

```gitignore
# Python
__pycache__/
*.py[cod]
venv/
.venv/

# Generated feature cache and model
features_X.npy
labels_y.npy
svm_speaker_model.pkl

# Local environment/editor files
.env
.vscode/
.idea/
```

If you choose to distribute a small dataset, remove or adjust the relevant dataset ignore rule only after confirming you have permission to share it.

## References

- [Whisper – OpenAI GitHub repository](https://github.com/openai/whisper)
- [librosa documentation](https://librosa.org/doc/latest/index.html)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [jiwer – Word Error Rate and related metrics](https://github.com/jitsi/jiwer)
- [NumPy documentation](https://numpy.org/doc/)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)

---

**Course:** CSE445 – Machine Learning  
**Project:** Speech Recognition and Speaker Classification System
