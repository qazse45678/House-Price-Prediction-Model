# House-Price-Prediction-Model

1. Define the question
2. Preprocessing
    * describe the data
    * train-test split
    * clean missing values
    * imputation
    * transform categorical value
3. Apply model  
    * create pipeline
    * cross validation to evaluate the model
    * find the best parameter for the model
4. Predict the test set
5. Add-ons: XGBoost

Time Spent: 1 week (2022.11)

# 1. Define the question
We have 79 variables to describe the features of houses, including location, age, area..., and of course, their prices. I'll use the dataset to train a **random forest and XGBoost model** to predict the price of each house. In the project, I'll also write a pipeline to connect multiple steps and make the process more simple.

# 2. Preprocessing
Before starting, let's connect the dataset from Kaggle to Jupyter notebook.
```
pip install opendatasets --quiet --upgrade
```
```
import opendatasets as od
download_url = 'https://www.kaggle.com/datasets/dansbecker/home-data-for-ml-course'

od.download(download_url)
```

Now, let's load the file.
```
import pandas as pd
from sklearn.model_selection import train_test_split

X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')
```

## Describe the data

1. what are the statstics metrics of the data?
```
data.describe()
```
<img width="1092" alt="image" src="https://user-images.githubusercontent.com/63503783/203723036-bd3b0bf4-d115-4276-9021-b41838c6b5b5.png">

2. What are the columns?
```
data.columns
```
<img width="629" alt="image" src="https://user-images.githubusercontent.com/63503783/203723219-261f879d-555b-4aeb-a17b-74bbcb844b86.png">

3. How many columns and rows are there in the dataset?
```
data.head()
```
<img width="1097" alt="image" src="https://user-images.githubusercontent.com/63503783/203723266-b253d9ea-c829-4c56-a811-5aea4eabf31f.png">

## Train test split
```
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
```

## Missing values
1. How many missing values are there for each feature?
```
print(X_train.shape)
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```
<img width="168" alt="image" src="https://user-images.githubusercontent.com/63503783/204991915-cc70961c-21bc-4cba-b8e4-7bf67b0058c3.png">

2. How many rows are in the training data?
```
num_rows = X_train.shape[0]
```

3. How many columns in the training data have missing values?
```
num_cols_with_missing = missing_val_count_by_column[missing_val_count_by_column > 0].shape[0]
```

4. How many missing entries are contained in all of the training data?
```
tot_missing = missing_val_count_by_column.sum()
```
<img width="608" alt="image" src="https://user-images.githubusercontent.com/63503783/204992774-f981a88f-405e-4f0b-8250-27a5bd88522a.png">

Now, let's drop columns with missing values.
First, get names of columns with missing values. And then drop those columns.
```
print(missing_val_count_by_column[missing_val_count_by_column > 0].index)

reduced_X_train = X_train.dropna(axis = 1)
reduced_X_valid = X_valid.dropna(axis = 1)
```

## Imputation
Now, I'll impute values with mean values to each columns which has missing values.
```
from sklearn.impute import SimpleImputer

imputing = SimpleImputer() # Your code here
imputed_X_train = pd.DataFrame(imputing.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputing.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
```

## transform categorical value
I'll use one-hot encoding method to transform categorical columns.
```
from sklearn.preprocessing import OneHotEncoder
```

Apply one-hot encoder to each column with categorical data
```
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
OH_X_train_col = pd.DataFrame(ohe.fit_transform(X_train[low_cardinality_cols]))
OH_X_valid_col = pd.DataFrame(ohe.transform(X_valid[low_cardinality_cols]))
```

One-hot encoding removed index; put it back
```
OH_X_train_col.index = X_train.index
OH_X_valid_col.index = X_valid.index
```

Remove categorical columns (will replace with one-hot encoding)
```
new_X_train = X_train.drop(X_train[object_cols], axis = 1)
new_X_valid = X_valid.drop(X_valid[object_cols], axis = 1)

OH_X_train = pd.concat([new_X_train, OH_X_train_col], axis = 1)
OH_X_valid = pd.concat([new_X_valid, OH_X_valid_col], axis = 1)
```

# 3. Apply model
## Pipeline and models
I'll create pipieline to combine last step (one-hot encoder) and model fitting.
First, import necessary libraries.
```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

And then, I'll preprocess numerical and categorical data separately. Here's how I set up the simple imputer for numerical data:
```
numerical_transformer = SimpleImputer(strategy='constant')
```

Here's the simple imputer setting for categorical data:
```
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

Let's combine the numerical and categorical data together.
```
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

It's time to define model and bundle the preprocessing steps and model together.
```
model = RandomForestRegressor(n_estimators=100, random_state=0)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])
```

Train and predict the model. Get MAE.
```
clf.fit(X_train, y_train)

preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
```
<img width="194" alt="image" src="https://user-images.githubusercontent.com/63503783/204994487-3e3d355e-c496-4b86-aa1d-a879aa96fdc7.png">

## Cross validation in the pipeline
```
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())
```
<img width="154" alt="image" src="https://user-images.githubusercontent.com/63503783/204995473-0e4affd7-a448-4977-a660-f14126561246.png">

## Find the best parameter for the model
To test the best parameter (number of tree) for random forest model, I'll list down three numbers and run them respectively. Let's see the performance (evaluating by MAE) with the chart below.
```
results = {i: get_score(i) for i in range(50, 450, 50)}
```
```
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()
```
<img width="403" alt="image" src="https://user-images.githubusercontent.com/63503783/204995833-6f380681-ebd7-42ba-aa30-d534543c3d55.png">

Here's the best parameter: 200
```
n_estimators_best = min(results, key = results.get)
```

# 4. XGBoost
Define, fit and predict the model.
```
from xgboost import XGBRegressor

my_model_1 = XGBRegressor(random_state = 0)

my_model_1.fit(X_train, y_train)

mae_1 = mean_absolute_error(predictions_1, y_valid)

print("Mean Absolute Error:" , mae_1)
```
<img width="324" alt="image" src="https://user-images.githubusercontent.com/63503783/204996819-db7f9c90-088a-4455-9581-d02a4f1c4d15.png">


Improve the model.
```
my_model_3 = XGBRegressor(n_estimators = 50, learning_rate = 0.05, random_state = 0)

my_model_3.fit(X_train, y_train)

predictions_3 = my_model_3.predict(X_valid)

mae_3 = mean_absolute_error(predictions_3, y_valid)

print("Mean Absolute Error:" , mae_3)

```
<img width="319" alt="image" src="https://user-images.githubusercontent.com/63503783/204996733-36f03afc-bdfa-4772-b875-875a24f4111a.png">
