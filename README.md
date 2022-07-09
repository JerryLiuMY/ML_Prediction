# ML_Prediction
<p align="left">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
</p>

## Data Information
<a href="https://drive.google.com/drive/folders/1Ha_viwpfKjF9OKcxVGTOlt8ZTtLkTJZo?usp=sharing" target="_blank">Repository</a> for the processed data. The features `x0-x797` has been demeaned and standardized daily. <a href="https://drive.google.com/drive/folders/1c1UBirLc1OhzoqG7O4ipa92F4YwEG43R?usp=sharing" target="_blank">Folder</a> for data information. <a href="__resources__/count.pdf" target="_blank">Number of stocks</a> in `X`, `y` and shared between `X & y` from *Jan 2010* to *Jun 2022*.

Dictionary of parameters: https://github.com/xiubooth/ML_Prediction/blob/main/params/params.py

## AutoGluon
<a href="https://drive.google.com/drive/folders/1elTNSDXkk9FjIR_8WyOvj1yvwk0LbNPM?usp=sharing" target="_blank">Repository</a> for the trained models, evaluation metrics and predictions. 
![alt text](./__resources__/autogluon/horizon=1.jpg?raw=true "Title")

**Correlation decay:** `0-6 days: 1.67%`, `6-12 days: 0.20%`, `12-18 days: -0.33%`, `18-24 days: 0.34%`, `24-30 days: 0.60%`, `30-36 days: 1.48%`, `36-42 days: 0.82%`, `42-48 days: 0.27%`, `48-54 days: 1.02%`, `54-60 days: 0.25%`

## TODO
- Ensemble learning: multiple ML models
- Ensemble learning: multiple horizons
- Ensemble based on particular by metric (e.g. pearsonr, r2, volatility)
- AutoGluon
  - Predictive accuracy
- Transformers 
  - Different modifications
  - Different window sizes
- Variable importance
- Random effect model
