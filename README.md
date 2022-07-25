# ML_Prediction
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Data Information
<a href="https://drive.google.com/drive/folders/19Ehs8GjoDrOkIO_Il7I6c3KRZ3R7JV1j?usp=sharing" target="_blank">Repository</a> for the processed data. <a href="https://drive.google.com/drive/folders/19ItSnvSwf3g6C0AQVetpFKySiHH9HBpi?usp=sharing">Folder</a> for data information. <a href="__resources__/count.pdf" target="_blank">Number of stocks</a> in `X`, `y` and shared between `X & y` from *Jan 2010* to *Jun 2022*. The features `x0 - x797` has been demeaned and standardized daily.

#### Tuning scheme
Dictionary of parameters: https://github.com/xiubooth/ML_Prediction/blob/main/params/params.py
- Increase training window
- Reduce testing window
- Obtain validation data from training window
- Predictive horizon

## AutoGluon
<a href="https://drive.google.com/drive/folders/1i5NAy-udEmN8g6bnyG1g76HTcsVvPd4J?usp=sharing" target="_blank">Repository</a> for the trained models, evaluation metrics and predictions. 
![alt text](./__resources__/autogluon/baseline.jpg?raw=true "Title")

**Parameters:** `Random Forest` excluded due to high computational costs (cannot be GPU accelerated) and uncompetitive performance. The benchmark parameters are `resampling=True`, `presets=medium_quality`, `train_window=240`, `valid_window=120`, `test_window=60`
- <a href="./__resources__/autogluon/resampling=False.pdf" target="_blank">Results</a> with `resampling=False`
- Results with `presets=high_quality`
- Results with `train_window=960`
- Results with `test_window=20`
- <a href="./__resources__/autogluon/horizon=2.pdf" target="_blank">Results</a> with `horizon=2`

## Transformer
![alt text](./__resources__/transformer/baseline.jpg?raw=true "Title")

**Parameters:** The benchmark parameters are `seq_len=5`, `nlayer=2`, `nhead=32`, `d_model=4096`, `dropout=0.05`, `epochs=5`, `lr=0.01`

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
