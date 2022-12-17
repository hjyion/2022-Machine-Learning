# 22-2 Machine Learning Team Project (FiveK)


### 1. Dataset
  We obtained our dataset from Kaggle.
  * [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. Requirements
  We used Google Colab with Google Drive for convenience. To access dataset on drive after download, code below will help you mount your drive to /content/drive/MyDrive directory.

```python
from google.colab import drive
import pandas as pd
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
```

  To run this project, Python and following libraries are required:
  * [Pandas](https://pandas.pydata.org/)
  * [Numpy](https://numpy.org/)
  * [Scikit-Learn](https://scikit-learn.org/stable/)
  * [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for Visualization

  Since Google Colab supports those libraries, we simply import them just as code below.
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
```

### 3. Models 
We used Sklearn libraries for...

* **XGBoost** (Our model)
* Logistic Regression, KNN, RandomForest, MLP for comparison

### 4. Performance Evaluation
* classification report
* confusion matrix

### 5. Results
