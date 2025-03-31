# OPTIFYX
### Car Price Prediction

### **1. Importing Libraries**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn import metrics
```
- `pandas` is used for data manipulation and analysis.
- `matplotlib.pyplot` and `seaborn` are used for visualization.
- `sklearn.model_selection` helps in splitting the dataset for training and testing.
- `LinearRegression` and `Lasso` are machine learning models used for price prediction.
- `metrics` is used to evaluate model performance.

---

### **2. Loading the Dataset**
```python
car_dataset = pd.read_csv('car data.csv')
```
- Reads the dataset from a CSV file into a Pandas DataFrame.

---

### **3. Exploring the Dataset**
```python
car_dataset.head()
```
- Displays the first 5 rows of the dataset.

```python
car_dataset.shape
```
- Shows the number of rows and columns in the dataset.

```python
car_dataset.info()
```
- Provides information about the dataset, including column names, data types, and non-null values.
Here’s a step-by-step explanation of the remaining code in your **Car Price Prediction** notebook:

---

## **4. Checking Categorical Data Distribution**
```python
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Selling_type.value_counts())
print(car_dataset.Transmission.value_counts())
```
- Displays the count of unique values in categorical columns like **Fuel_Type**, **Selling_type**, and **Transmission**.

---

## **5. Encoding Categorical Data**
```python
# Encoding the "Fuel_Type" column
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)

# Encoding the "Selling_type" column
car_dataset.replace({'Selling_type': {'Dealer': 0, 'Individual': 1}}, inplace=True)

# Encoding the "Transmission" column
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
```
- Converts categorical variables into numerical values:
  - **Fuel_Type**: Petrol → 0, Diesel → 1, CNG → 2  
  - **Selling_type**: Dealer → 0, Individual → 1  
  - **Transmission**: Manual → 0, Automatic → 1  

```python
car_dataset.head()
```
- Displays the modified dataset after encoding.

---

## **6. Splitting Features and Target Variable**
```python
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']
```
- **X**: Independent variables (features) after dropping **Car_Name** and **Selling_Price**.
- **Y**: Dependent variable (**Selling_Price**) to be predicted.

```python
print(X)
print(Y)
```
- Displays feature matrix (`X`) and target variable (`Y`).

---

## **7. Splitting Training and Testing Data**
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
```
- Splits the dataset into **90% training** and **10% testing**.

---

## **8. Model Training - Linear Regression**
```python
# Loading Linear Regression model
lin_reg_model = LinearRegression()
```
- Creates a **Linear Regression** model.

```python
lin_reg_model.fit(X_train, Y_train)
```
- Trains the model using training data.

---

## **9. Model Evaluation - Linear Regression**
```python
# Prediction on Training Data
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error (Training Data): ", error_score)
```
- Predicts prices on training data.
- Evaluates model accuracy using **R² score**.

```python
# Prediction on Test Data
test_data_prediction = lin_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error (Test Data): ", error_score)
```
- Predicts on **test data** and calculates **R² score**.

---

## **10. Model Training - Lasso Regression**
```python
# Loading Lasso Regression model
lasso_reg_model = Lasso()
```
- Creates a **Lasso Regression** model.

```python
lasso_reg_model.fit(X_train, Y_train)
```
- Trains the **Lasso Regression** model.

---

## **11. Model Evaluation - Lasso Regression**
```python
# Prediction on Training Data
training_data_prediction = lasso_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error (Training Data - Lasso): ", error_score)
```
- Predicts on **training data** using **Lasso Regression**.
- Calculates **R² score**.

```python
# Prediction on Test Data
test_data_prediction = lasso_reg_model.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error (Test Data - Lasso): ", error_score)
```
- Predicts on **test data** using **Lasso Regression**.
- Calculates **R² score**.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Email Spam Detection


### **1. Importing Necessary Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
- `numpy` and `pandas`: Used for data handling and manipulation.
- `train_test_split`: Splits the dataset into training and testing sets.
- `TfidfVectorizer`: Converts text data into numerical form.
- `LogisticRegression`: The machine learning model used for classification.
- `accuracy_score`: Evaluates the model’s accuracy.

---

### **2. Loading the Dataset**
```python
df = pd.read_csv('spam.csv', encoding="latin-1")
```
- Reads the **spam.csv** file into a Pandas DataFrame.
- The dataset likely contains emails labeled as spam or ham (not spam).

---

### **3. Displaying the Data**
```python
df
```
- Displays the first few rows of the dataset.

---

### **4. Removing Unnecessary Columns**
```python
dt = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
```
- Drops extra columns that are not required for spam detection.

```python
dt
```
- Displays the cleaned dataset.

---

## **5. Checking Dataset Shape**
```python
dt.shape
```
- Displays the number of rows and columns in the cleaned dataset.

---

## **6. Handling Missing Values**
```python
data = dt.where((pd.notnull(dt)), '')
```
- Replaces any `NaN` (missing) values with an empty string.

```python
data.head(10)
```
- Displays the first 10 rows of the processed dataset.

```python
data.shape
```
- Confirms the shape of the dataset after handling missing values.

---

## **7. Encoding the Target Column**
```python
data.loc[data['v1'] == 'spam', 'v1'] = 0
data.loc[data['v1'] == 'ham', 'v1'] = 1
```
- Converts categorical labels into numeric values:
  - **Spam** → `0`
  - **Ham (Not Spam)** → `1`

---

## **8. Splitting Features and Labels**
```python
X = data['v2']  # Feature (Email text)
Y = data['v1']  # Target (Spam or Ham)
```
- **X** contains the email messages.
- **Y** contains labels (0 for spam, 1 for ham).

```python
print(X)
print(Y)
```
- Displays feature data (`X`) and labels (`Y`).

---

## **9. Splitting the Dataset into Training & Testing Sets**
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
```
- Splits the dataset:
  - **80% Training Data**
  - **20% Testing Data**

```python
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)
```
- Displays the shape of training and testing datasets.

---

## **10. Feature Extraction using TF-IDF Vectorizer**
```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
```
- **TF-IDF Vectorizer** converts text data into numerical values:
  - `min_df=1`: Ignores words that appear in very few documents.
  - `stop_words='english'`: Removes common English stop words.
  - `lowercase=True`: Converts all text to lowercase.
- The transformed **X_train_features** and **X_test_features** are ready for model training.

```python
print(X_train_features)
```
- Displays the numerical representation of training features.

---

## **11. Training the Logistic Regression Model**
```python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```
- **Logistic Regression** is used to classify emails as spam or not.
- The model is trained using **X_train_features** and **Y_train**.

---

## **12. Model Evaluation on Training Data**
```python
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print("Accuracy on Training Data: ", accuracy_on_training_data)
```
- Predicts labels on training data.
- Computes **accuracy score** for training data.

---

## **13. Model Evaluation on Testing Data**
```python
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Accuracy on Test Data: ", accuracy_on_test_data)
```
- Predicts labels on test data.
- Computes **accuracy score** for test data.

---

## **14. Making Predictions on New Data**
```python
input_mail = ["Win a free iPhone now! Click here to claim your prize."]
input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
print(prediction)

if prediction[0] == 0:
    print("Spam Email")
else:
    print("Not Spam Email")
```
- Example email **"Win a free iPhone now!"** is vectorized using TF-IDF.
- The trained model predicts whether it's **spam (0) or not (1)**.
- Displays the prediction result.

---
