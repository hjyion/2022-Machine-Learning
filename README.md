# 22-2 Machine Learning Team Project (FiveK)
![11](https://user-images.githubusercontent.com/80759204/208238456-7ee65f32-ecff-4525-bbf7-7369a9aa5c0c.JPG)

This project is about detecting fraud credit card transactions by using proper machine learning algorithm. We chose XGBoost as our model and decided to apply oversampling technique named SMOTE. However, our experiment showed that applying SMOTE on XGBoost leads to worse performance in terms of precision. Thus, we tried to discover the reason why SMOTE did not improve the performance of XGBoost with our dataset. To reach our conclusion, we designed several experimental scenarios such as changing base rate of test set and comparing some evaluation metrics with other models. Our experimental results can be found below. Thank you!

### 1. Dataset

  We obtained our dataset from Kaggle.
  * [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. Requirements
  We used Google Colab with Google Drive for convenience. To access dataset on drive after download, code below will help you mount your drive to **/content/drive/MyDrive** directory.

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
We used Sklearn libraries for models below.

* **XGBoost** (Our model)
* Logistic Regression
* KNN (n_neighbors = 3)
* RandomForest
* MLP

### 4. Results

* XGBoost 

Obtained ROC curve and Precision-Recall curve

![14](https://user-images.githubusercontent.com/80759204/208238636-b9b5bd12-190c-451a-b23e-4bd12223fde9.JPG)


* Changing base rate of test set (1:1, 1:10, 1:100, 1:577.87(original))

Obtained F1 score and Precision for 4 different models (Logistic Regression, Random Forest, MLP, XGBoost(Ours))

![15](https://user-images.githubusercontent.com/80759204/208238707-607dfec5-295a-4b5e-9996-ea1ef4d070c3.JPG)




* Comparision with other models

<img src="https://user-images.githubusercontent.com/80759204/208238764-be4e4b37-b789-48d7-be0a-299ca0263de1.JPG"  width="500" height="200">

### 5. Interpretations

* Why SMOTE didn't improve the performance?

We found that SMOTE uses Euclidean distance for K-NN to create new samples, which is not good for high-dimensional space. Without variable selection, K-NN classification tends to be biased towards the minority class so that FP increases.

![17](https://user-images.githubusercontent.com/80759204/208238892-881c6790-0d40-4b6c-bcf2-6bc48ad59523.JPG)

_Reference : L. Lusa and R. Blagus, “Evaluation of SMOTE for high-dimensional class-imbalanced microarray data,” in Proc. 11th Int. Conf. Mach. Learn. Appl., vol. 2, 2012_

### 6. Discussion

So, does oversampling really work on our dataset in other ways?

![18](https://user-images.githubusercontent.com/80759204/208239117-6a99bfb3-202e-418b-85a4-79226e1950d9.JPG)

To answer this question, we can try other methods such as:
  1. Perform feature selection
  2. Try other oversampling techniques such as ADASYN, borderline SMOTE, etc.
  3. Try hybrid method of oversampling and undersampling to avoid the prediction to be biased in either class.
