# ML_Prediction
<p align="left">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-v3-brightgreen.svg"
            alt="python"></a> &nbsp;
</p>

## Data Information
<a href="https://drive.google.com/drive/folders/1Ha_viwpfKjF9OKcxVGTOlt8ZTtLkTJZo?usp=sharing" target="_blank">Repository</a> for the processed data. <a href="https://drive.google.com/drive/folders/1c1UBirLc1OhzoqG7O4ipa92F4YwEG43R?usp=sharing" target="_blank">Folder</a> for data information. <a href="__resources__/count.pdf" target="_blank">Number of stocks</a> in `X`, `y` and shared between `X & y` from *Jan 2010* to *Jun 2022*. The features `x0 - x797` has been demeaned and standardized daily.

#### Tuning scheme
Dictionary of parameters: https://github.com/xiubooth/ML_Prediction/blob/main/params/params.py
- Increase training window
- Reduce testing window
- Obtain validation data from training window
- Predictive horizon

## AutoGluon
<a href="https://drive.google.com/drive/folders/1elTNSDXkk9FjIR_8WyOvj1yvwk0LbNPM?usp=sharing" target="_blank">Repository</a> for the trained models, evaluation metrics and predictions. 
![alt text](./__resources__/autogluon/baseline.jpg?raw=true "Title")

**Parameters:** `Random Forest` excluded due to high computational costs (cannot be GPU accelerated) and uncompetitive performance. The benchmark parameters are `presets=medium_quality`, `train_window=240`, `valid_window=120`, `test_window=60`
- <a href="./__resources__/autogluon/resampling.pdf" target="_blank">Results</a> with `resampling=True`
- Results with `presets=high_quality`
- Results with `train_window=960`
- Results with `test_window=20`
- Results with `horizon=2`

## TODO
- Ensemble learning: multiple ML models
- Ensemble learning: based on particular by metric (e.g. pearsonr, r2, volatility)
- Transformers 
  - Different modifications
  - Different window sizes
- Use correlation as loss
- Change IC to portfolio
- Kaggle reference
- Variable importance
- Random effect model
