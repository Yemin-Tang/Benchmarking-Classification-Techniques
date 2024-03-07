#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DEA CBR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv('BBC new classification data.csv')

# Separate features and target variable
features = data[['liabilities_to_assets', 'total_debt_to_assets',
                 'current_assets_to_liabilities', 'net_income_to_assets',
                 'cash_flows_to_assets', 'working_capital_to_assets',
                 'retained_earning_to_assets', 'operating_earning_to_assets',
                 'sales_to_total_asset']]
target = data['bankruptcy_status']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a custom similarity measure
def similarity_measure(case1, case2):
    return sum((case1 - case2) ** 2)  # Euclidean distance for simplicity

# Function to retrieve k most similar cases from the case base
def retrieve_cases(query_case, case_base, k=5):
    similarities = [similarity_measure(query_case, case['features']) for case in case_base]
    sorted_indices = sorted(range(len(similarities)), key=lambda x: similarities[x])
    return [case_base[i] for i in sorted_indices[:k]]

# Function to predict based on the most similar cases
def predict(query_case, case_base, k=5):
    similar_cases = retrieve_cases(query_case, case_base, k)
    predictions = [case['bankruptcy'] for case in similar_cases]
    return max(set(predictions), key=predictions.count)

# Create a case base using the training data
case_base = [{'features': X_train.iloc[i].values, 'bankruptcy': y_train.iloc[i]} for i in range(len(X_train))]

# Make predictions on the testing set
predictions = [predict(X_test.iloc[i].values, case_base) for i in range(len(X_test))]

# Evaluate the model on the testing set
accuracy = sum(predictions[i] == y_test.iloc[i] for i in range(len(y_test))) / len(y_test)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)


# In[2]:


#DEA-DT-CBR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv('BBC new classification data.csv')

# Separate features and target variable
features = data[['liabilities_to_assets', 'net_income_to_assets']]
target = data['bankruptcy_status']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Combine features and target for the entire dataset
X = features
y = target

# Define a custom similarity measure
def similarity_measure(case1, case2):
    return sum((case1 - case2) ** 2)  # Euclidean distance for simplicity

# Function to retrieve k most similar cases from the case base
def retrieve_cases(query_case, case_base, k=5):
    similarities = [similarity_measure(query_case, case['features']) for case in case_base]
    sorted_indices = sorted(range(len(similarities)), key=lambda x: similarities[x])
    return [case_base[i] for i in sorted_indices[:k]]

# Function to predict based on the most similar cases
def predict(query_case, case_base, k=5):
    similar_cases = retrieve_cases(query_case, case_base, k)
    predictions = [case['bankruptcy'] for case in similar_cases]
    return max(set(predictions), key=predictions.count)

# Create a case base using the training data
case_base = [{'features': X_train.iloc[i].values, 'bankruptcy': y_train.iloc[i]} for i in range(len(X_train))]

# Make predictions on the testing set
predictions = [predict(X_test.iloc[i].values, case_base) for i in range(len(X_test))]

# Evaluate the model
accuracy = sum(predictions[i] == y_test.iloc[i] for i in range(len(y_test))) / len(y_test)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)


# In[4]:


#DT-CBR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv('DT_CBR.csv')

# Separate features and target variable
features = data[['retained_earning_to_assets', 'total_debt_to_assets', 'net_income_to_assets']]
target = data['bankruptcy_status']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a custom similarity measure (Euclidean distance)
def similarity_measure(case1, case2):
    return sum((case1 - case2) ** 2)

# Function to retrieve k most similar cases from the case base
def retrieve_cases(query_case, case_base, k=5):
    similarities = [similarity_measure(query_case, case['features']) for case in case_base]
    sorted_indices = sorted(range(len(similarities)), key=lambda x: similarities[x])
    return [case_base[i] for i in sorted_indices[:k]]

# Function to predict based on the most similar cases
def predict(query_case, case_base, k=5):
    similar_cases = retrieve_cases(query_case, case_base, k)
    predictions = [case['bankruptcy'] for case in similar_cases]
    return max(set(predictions), key=predictions.count)

# Create a case base using the training data
case_base = [{'features': X_train.iloc[i].values, 'bankruptcy': y_train.iloc[i]} for i in range(len(X_train))]

# Make predictions on the testing set
predictions = [predict(X_test.iloc[i].values, case_base) for i in range(len(X_test))]

# Evaluate the model on the testing set
accuracy = sum(predictions[i] == y_test.iloc[i] for i in range(len(y_test))) / len(y_test)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)


# In[4]:


#CBR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read data
data = pd.read_csv('ourdataset.csv')

# Separate features and target variable
features = data[['liabilities_to_assets', 'total_debt_to_assets',
                 'current_assets_to_liabilities', 'net_income_to_assets',
                 'cash_flows_to_assets', 'working_capital_to_assets',
                 'retained_earning_to_assets', 'operating_earning_to_assets',
                 'sales_to_total_asset']]
target = data['bankruptcy_status']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a custom similarity measure
def similarity_measure(case1, case2):
    return sum((case1 - case2) ** 2)  # Euclidean distance for simplicity

# Function to retrieve k most similar cases from the case base
def retrieve_cases(query_case, case_base, k=5):
    similarities = [similarity_measure(query_case, case['features']) for case in case_base]
    sorted_indices = sorted(range(len(similarities)), key=lambda x: similarities[x])
    return [case_base[i] for i in sorted_indices[:k]]

# Function to predict based on the most similar cases
def predict(query_case, case_base, k=5):
    similar_cases = retrieve_cases(query_case, case_base, k)
    predictions = [case['bankruptcy'] for case in similar_cases]
    return max(set(predictions), key=predictions.count)

# Create a case base using the training data
case_base = [{'features': X_train.iloc[i].values, 'bankruptcy': y_train.iloc[i]} for i in range(len(X_train))]

# Make predictions on the testing set
predictions = [predict(X_test.iloc[i].values, case_base) for i in range(len(X_test))]

# Evaluate the model on the testing set
accuracy = sum(predictions[i] == y_test.iloc[i] for i in range(len(y_test))) / len(y_test)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)


# In[ ]:




