# House-Price-Prediction-Model

```pip install opendatasets --quiet --upgrade```
```import opendatasets as od
download_url = 'https://www.kaggle.com/datasets/dansbecker/home-data-for-ml-course'

od.download(download_url)```

# Load the file
```import pandas as pd

file_path = './home-data-for-ml-course/train.csv'
data = pd.read_csv(file_path)```

# Understand the summary of the file

## 1. what are the statstics metrics of the data?

## 2. How many/ what are the columns of the data?
```data.describe()```
