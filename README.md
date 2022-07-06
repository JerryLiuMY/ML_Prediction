# ML_Prediction
<p align="left">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
</p>

## Data Information
<a href="https://drive.google.com/drive/folders/1Ha_viwpfKjF9OKcxVGTOlt8ZTtLkTJZo?usp=sharing" target="_blank">Repository</a> for the processed data. <a href="https://drive.google.com/drive/folders/1c1UBirLc1OhzoqG7O4ipa92F4YwEG43R?usp=sharing" target="_blank">Folder</a> for data information. Number of stocks in `X`, `y` and shared between `X & y` from *Jan 2010* to *Jun 2022*.

[//]: # (![alt text]&#40;./__resources__/count.jpg?raw=true "Title"&#41;)

Dictionary of parameters: https://github.com/xiubooth/ML_Prediction/blob/main/params/params.py

## AutoGluon
<a href="https://drive.google.com/drive/folders/1elTNSDXkk9FjIR_8WyOvj1yvwk0LbNPM?usp=sharing" target="_blank">Repository</a> for the trained models, evaluation metrics and predictions. 

![alt text](./__resources__/autogluon.jpg?raw=true "Title")

**Correlation Decay** `0-6 days: 1.00%`, `6-12 days: 0.00%`, `12-18 days: -0.81%`, `18-24 days: 0.04%`, `24-30 days: 0.35%`, `30-36 days: 1.21%`, `36-42 days: 0.40%`, `42-48 days: 0.27%`, `48-54 days: 1.34%`, `54-60 days: 1.00%`

## TODO
- Ensemble learning: multiple ML models
- Ensemble based on particular by metric (e.g. pearsonr, r2, volatility)
- Multiple ML predictive horizons
- Transformers 
  - Different modifications
  - Different window sizes
- Backtest / cumulative correlation
- Variable importance
- Random effect model
