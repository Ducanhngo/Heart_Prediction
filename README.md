# Stacking Model for Classification Using Supervised Learning Algorithms

This project demonstrates how to build a **stacking model** for a classification task using multiple **supervised learning algorithms**. The base models include **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, **Gradient Boosting**, **AdaBoost**, and **Support Vector Machine (SVM)**. **XGBoost** is used as the meta-model in this stacking ensemble.

## Acknowledgments
I want to express my heartfelt thanks to [AI VIET NAM](https://aivietnam.edu.vn/) for their incredible support and guidance throughout this project. Their assistance has been invaluable in making this project a success.

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithms Used](#algorithms-used# Stacking Model for Classification Using Supervised Learning Algorithms

This project demonstrates how to build a **stacking model** for a classification task using multiple **supervised learning algorithms**. The base models include **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, **Gradient Boosting**, **AdaBoost**, and **Support Vector Machine (SVM)**. **XGBoost** is used as the meta-model in this stacking ensemble.
Dataset --> [Click here](https://drive.google.com/file/d/1zOj808OstnkaWlltM4qKNjjT3iT3yeMN/view)
## Table of Contents
1. [Introduction](#introduction)
2. [Algorithms Used](#algorithms-used)
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting
   - AdaBoost
   - Support Vector Machine (SVM)
   - XGBoost (Meta-Model)
3. [Stacking Model](#stacking-model)
4. [Implementation](#implementation)
   - Data Preprocessing
   - Model Training
   - Performance Evaluation
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

This project aims to build a robust classification model by combining multiple supervised learning algorithms into a **stacking ensemble**. Stacking allows us to improve the overall predictive performance by leveraging the strengths of various models. In this implementation:
- Base models: Decision Tree, Random Forest, KNN, Gradient Boosting, AdaBoost, and SVM.
- Meta-model: XGBoost is used to learn from the outputs of the base models.
![Models](https://github.com/Ducanhngo/Heart_Prediction/blob/main/Models.png)
## Algorithms Used

### Decision Tree
A **Decision Tree** classifier is a tree-like model that makes decisions based on feature values. It is prone to overfitting, but we mitigate this by combining it with other models in the stacking ensemble.
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2)
```

### Random Forest
**Random Forest** is an ensemble method that builds multiple decision trees and merges them together to reduce variance and improve accuracy.
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_split=2, n_estimators = 10, random_state=42)
```

### K-Nearest Neighbors (KNN)
The **KNN** algorithm classifies instances based on the majority class among its k nearest neighbors in the feature space.
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric = 'minkowski')
```

### Gradient Boosting
**Gradient Boosting** builds models sequentially, where each model corrects the errors made by the previous one, focusing on the difficult cases.
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
```

### AdaBoost
**AdaBoost** is another boosting algorithm that adjusts weights on misclassified instances and focuses on difficult-to-classify samples.
```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
```

### Support Vector Machine (SVM)
**SVM** is a powerful classifier that works well for both linear and non-linear data. It finds the optimal hyperplane that best separates the classes.
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=42)
```

### XGBoost (Meta-Model)
**XGBoost** is used as the meta-model in the stacking ensemble. It is an efficient implementation of gradient boosting that is highly performant for structured data.
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, random_state=42)
```

## Stacking Model

The stacking ensemble combines all the base models and uses the predictions of these models as inputs to the meta-model. This way, the meta-model (XGBoost) can learn from the collective predictions of the base models.

```python
from sklearn.ensemble import StackingClassifier

# Define the base models
base_models = [
    ('dt', DecisionTreeClassifier(criterion='entropy', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

# Define the meta-model (XGBoost)
meta_model = XGBClassifier(n_estimators=100, random_state=42)

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the model
stacking_model.fit(X_train, y_train)
```

## Implementation

### Data Preprocessing
The dataset is split into training and test sets using `train_test_split`, and the features and target variables are extracted from the DataFrame.

```python
from sklearn.model_selection import train_test_split

# Assume df is your DataFrame
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training
The stacking model is trained on the training set, and predictions are made for both the training and test sets.

```python
# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)
```

### Performance Evaluation
Performance is evaluated using accuracy and confusion matrix to measure the correctness of the model on the training and test sets.

```python
from sklearn.metrics import confusion_matrix, accuracy_score

# Confusion matrix and accuracy for the training set
cm_train = confusion_matrix(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)

# Confusion matrix and accuracy for the test set
cm_test = confusion_matrix(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f'Accuracy for training set: {accuracy_train}')
print(f'Accuracy for test set: {accuracy_test}')
```

## Results
After training and testing the stacking model, the results show improved accuracy and generalization due to the combination of different models' strengths.

- **Training Accuracy**: X%
- **Test Accuracy**: Y%

## Conclusion
By combining multiple supervised learning algorithms in a stacking ensemble, we are able to improve the predictive performance of our model. The meta-model (XGBoost) successfully learns from the base models' outputs, leading to better accuracy on both the training and test sets.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---


   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting
   - AdaBoost
   - Support Vector Machine (SVM)
   - XGBoost (Meta-Model)
3. [Stacking Model](#stacking-model)
4. [Implementation](#implementation)
   - Data Preprocessing
   - Model Training
   - Performance Evaluation
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction

This project aims to build a robust classification model by combining multiple supervised learning algorithms into a **stacking ensemble**. Stacking allows us to improve the overall predictive performance by leveraging the strengths of multiple models. In this implementation:
- Base models: Decision Tree, Random Forest, KNN, Gradient Boosting, AdaBoost, and SVM.
- Meta-model: XGBoost is used to learn from the outputs of the base models.

## Algorithms Used

### Decision Tree
A **Decision Tree** classifier is a tree-like model that makes decisions based on feature values. It is prone to overfitting, but we mitigate this by combining it with other models in the stacking ensemble.
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
```

### Random Forest
**Random Forest** is an ensemble method that builds multiple decision trees and merges them together to reduce variance and improve accuracy.
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
```

### K-Nearest Neighbors (KNN)
The **KNN** algorithm classifies instances based on the majority class among its k nearest neighbors in the feature space.
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
```

### Gradient Boosting
**Gradient Boosting** builds models sequentially, where each model corrects the errors made by the previous one, focusing on the difficult cases.
```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
```

### AdaBoost
**AdaBoost** is another boosting algorithm that adjusts weights on misclassified instances and focuses on difficult-to-classify samples.
```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
```

### Support Vector Machine (SVM)
**SVM** is a powerful classifier that works well for both linear and non-linear data. It finds the optimal hyperplane that best separates the classes.
```python
from sklearn.svm import SVC
svm = SVC(kernel='rbf', probability=True, random_state=42)
```

### XGBoost (Meta-Model)
**XGBoost** is used as the meta-model in the stacking ensemble. It is an efficient implementation of gradient boosting that is highly performant for structured data.
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, random_state=42)
```

## Stacking Model

The stacking ensemble combines all the base models and uses the predictions of these models as inputs to the meta-model. This way, the meta-model (XGBoost) can learn from the collective predictions of the base models.

```python
from sklearn.ensemble import StackingClassifier

# Define the base models
base_models = [
    ('dt', DecisionTreeClassifier(criterion='entropy', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

# Define the meta-model (XGBoost)
meta_model = XGBClassifier(n_estimators=100, random_state=42)

# Create the stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train the model
stacking_model.fit(X_train, y_train)
```

## Implementation

### Data Preprocessing
The dataset is split into training and test sets using `train_test_split`, and the features and target variables are extracted from the DataFrame.

```python
from sklearn.model_selection import train_test_split

# Assume df is your DataFrame
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training
The stacking model is trained on the training set, and predictions are made for both the training and test sets.

```python
# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)
```

### Performance Evaluation
Performance is evaluated using accuracy and confusion matrix to measure the correctness of the model on the training and test sets.

```python
from sklearn.metrics import confusion_matrix, accuracy_score

# Confusion matrix and accuracy for the training set
cm_train = confusion_matrix(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)

# Confusion matrix and accuracy for the test set
cm_test = confusion_matrix(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f'Accuracy for training set: {accuracy_train}')
print(f'Accuracy for test set: {accuracy_test}')
```

## Results
After training and testing the stacking model, the results show improved accuracy and generalization due to the combination of different models' strengths.

- **Training Accuracy**: X%
- **Test Accuracy**: Y%

## Conclusion
By combining multiple supervised learning algorithms in a stacking ensemble, we are able to improve the predictive performance of our model. The meta-model (XGBoost) successfully learns from the base models' outputs, leading to better accuracy on both the training and test sets.

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

