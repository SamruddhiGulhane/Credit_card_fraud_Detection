# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %% [markdown]
# loading dataset to pandas dataset

# %%
credit_card_data=pd.read_csv(r'C:\Users\91988\Downloads\archive (2).zip')

# %%
credit_card_data.head()

# %%
credit_card_data.tail()

# %%
credit_card_data.info()

# %%
credit_card_data.isnull().sum()

# %% [markdown]
# Distribution of ligit transaction and fraudulent 

# %%
credit_card_data['Class'].value_counts()

# %%
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


# %%
print(legit.shape)
print(fraud.shape)

# %% [markdown]
# 0=normal transaction
# 1=fraudlent transaction

# %%
fraud.Amount.describe()

# %%
#compare the values for both transactions
credit_card_data.groupby('Class').mean()

# %%


# %%
legit_sample=legit.sample(n=492)

# %%
new_dataset=pd.concat([legit_sample,fraud],axis=0)

# %%
new_dataset.head()

# %%
new_dataset.tail()

# %%
new_dataset['Class'].value_counts()

# %%
new_dataset.groupby('Class').mean()

# %% [markdown]
# spliting dataset into features and Targets

# %%
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']

# %%
print(Y)

# %% [markdown]
# split the data traning data & Testing data

# %%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# %%
print(X.shape,X_train.shape,X_test.shape)

# %%
model=LogisticRegression()

# %% [markdown]
# training the logistic Regerssion model with traning data

# %%
model.fit(X_train,Y_train)

# %% [markdown]
# accuracy on traning data

# %%
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

# %%
print('Accuracy on Training data:',training_data_accuracy)

# %% [markdown]
# accuracy on test data

# %%
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

# %%
print('accuracy_score on test data:',test_data_accuracy)


