# House-Price-Prediction-Model

```
pip install opendatasets --quiet --upgrade
```
```
import opendatasets as od
download_url = 'https://www.kaggle.com/datasets/dansbecker/home-data-for-ml-course'

od.download(download_url)
```

# Load the file
```
import pandas as pd

file_path = './home-data-for-ml-course/train.csv'
data = pd.read_csv(file_path)
```

# Understand the summary of the file

## 1. what are the statstics metrics of the data?

## 2. How many/ what are the columns of the data?

```
data.describe()
```
<img width="1092" alt="image" src="https://user-images.githubusercontent.com/63503783/203723036-bd3b0bf4-d115-4276-9021-b41838c6b5b5.png">

```
data.columns
```
<img width="629" alt="image" src="https://user-images.githubusercontent.com/63503783/203723219-261f879d-555b-4aeb-a17b-74bbcb844b86.png">

```
data.head()
```
<img width="1097" alt="image" src="https://user-images.githubusercontent.com/63503783/203723266-b253d9ea-c829-4c56-a811-5aea4eabf31f.png">

# Clean out invalid data
```
data.dropna()
```

# Specify prediction target and features (X, y). Train the model and predict
```
y = data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3)

from sklearn.tree import DecisionTreeRegressor

data_model = DecisionTreeRegressor(random_state = 1)
data_model.fit(train_X, train_y)

prediction_y = data_model.predict(test_X)
print(prediction_y)
```
<img width="579" alt="image" src="https://user-images.githubusercontent.com/63503783/203723681-1df0fba5-9ec2-4ca1-b3ce-54708877f99b.png">

# Calculate the Mean Absolute Error in Validation Data
```
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(prediction_y, test_y)
print(mae)
```
<img width="156" alt="image" src="https://user-images.githubusercontent.com/63503783/203723776-2269b4f4-d908-4d39-8f32-589dc3c83a32.png">

# Find the optimal tree leaves based on lowest MAE
```
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    
    data_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    data_model.fit(train_X, train_y)
    prediction_y = data_model.predict(test_X)
    mae = mean_absolute_error(test_y, prediction_y)
    
    return mae

leaves = [5, 25, 50, 100, 250, 500]
score = {i: get_mae(i, train_X, test_X, train_y, test_y) for i in leaves}
best_leaf = min(score, key = score.get)

final_model = DecisionTreeRegressor(max_leaf_nodes = best_leaf, random_state=0)
```

