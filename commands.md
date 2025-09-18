# Sentiment Analysis MLOps – Full Commands & Config

This file tracks all steps, commands, and ignore rules for the project.

---

## 1. Setup

### Create virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt


## 2. Data setup

Download IMDB dataset and extract them into data/raw/


## 3. Preprocessing and dataset building
```powershell
python src/preprocess.py
python src/build_dataset.py


## 4. Training and evaluation - trains with features 5000 and 1000 and logs metrics and artifacts to mlruns/
```powershell
python src/train.py


## 5. Inference(sample testing)
```powershell
python src/infer.py
## Example output :
Review: I absolutely loved this movie!
Predicted: Positive (Probabilities: [0.08 0.92])


## 6. MLFlow UI - http://127.0.0.1:5000
```powershell
mlflow ui