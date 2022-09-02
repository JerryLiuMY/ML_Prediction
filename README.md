# ML_Prediction
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Data Information
Repositories for the data: <a href="https://drive.google.com/drive/folders/1mnn6lLWGSudLeWrNvC8aSzwGEvCxG3V1?usp=sharing">raw_data</a>, <a href="https://drive.google.com/drive/folders/15_uPaEnbewDlTv-vlYVxQicbdF9cwyrR?usp=sharing">set1_data</a>, <a href="https://drive.google.com/drive/folders/17-T_YKH8DINJAQ4L5S3pk9CVjQWZCYzT?usp=sharing">set1_data2</a>, <a href="https://drive.google.com/drive/folders/13BR955HTHuEzNB8v1Irf8Yu-mTJo9IHi?usp=sharing">set2_data</a>, <a href="https://drive.google.com/drive/folders/1ST6G7gWZe0eqS2mlugXH99OD1lQo6baY?usp=sharing">set2_data2</a>. <a href="https://drive.google.com/drive/folders/17KIh6DNflulVJ2Ivar5TkVY7UJ44nlQJ?usp=sharing">Folder</a> for data information. <a href="__resources__/exploration.pdf" target="_blank">Summary</a> of a) the number of stocks in `X`, `y` and shared between `X & y` b) missing pattern in `X` from *Jan 2010* to *Jun 2022*. The features `x0 - x797` has been demeaned and standardized daily.

#### Tuning scheme
Dictionary of parameters: https://github.com/xiubooth/ML_Prediction/blob/main/params/params.py
- Increase training window
- Reduce testing window
- Obtain validation data from training window
- Predictive horizon

## AutoGluon
<a href="https://drive.google.com/drive/folders/174vaHteTtcNFIO9xRVcpFESngPQXJPBE?usp=sharing" target="_blank">Repository</a> for the trained models, evaluation metrics and predictions. 
![alt text](./__resources__/autogluon/baseline.jpg?raw=true "Title")

**Parameters:** `Random Forest` excluded due to high computational costs (cannot be GPU accelerated) and uncompetitive performance. The benchmark parameters are `resample=True`, `imputation=default`, `presets=medium_quality`, `train_window=240`, `valid_window=120`, `test_window=60`, `horizon=1`
- <a href="./__resources__/autogluon/resample=False.pdf" target="_blank">Results</a> with `resample=False`
- <a href="./__resources__/autogluon/imputation=zero.pdf" target="_blank">Results</a> with `imputation=zero`
- <a href="./__resources__/autogluon/imputation=drop.pdf" target="_blank">Results</a> with `imputation=drop`
- <a href="./__resources__/autogluon/presets=high_quality.pdf" target="_blank">Results</a> with `presets=high_quality`
- <a href="./__resources__/autogluon/train_window=960.pdf" target="_blank">Results</a> with `train_window=960`
- <a href="./__resources__/autogluon/test_window=20.pdf" target="_blank">Results</a> with `test_window=20`
- <a href="./__resources__/autogluon/horizon=2.pdf" target="_blank">Results</a> with `horizon=2`
- <a href="./__resources__/autogluon/DATA_TYPE=set1_data2.pdf" target="_blank">Results</a> with `DATA_TYPE=set1_data2`
- Results with `DATA_TYPE=set2_data`


## Transformer
![alt text](./__resources__/transformer/baseline.jpg?raw=true "Title")

**Parameters:** The benchmark parameters are `seq_len=5`, `nlayer=2`, `nhead=8`, `d_model=4096`, `dropout=0.1`, `epochs=7`, `lr=0.05`

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
