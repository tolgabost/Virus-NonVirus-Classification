# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:10:48 2023

@author: PC
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset.csv")

pd.DataFrame(df)

df.describe()

df.info()

print(df.isVirus.value_counts()) #667 virus - 1332 non-virus
sns.countplot(df.isVirus)

df.isVirus.isna().sum()
df.isnull().sum() #How many missing values do we have between features

df.isVirus = df.isVirus.replace({True: 1, False: 0})

df.hist(figsize=(20,20))
sns.displot(df.feature_1)
sns.boxplot(df.feature_1)

corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot=True, fmt=".2f")

plt.scatter(df.feature_1,df.feature_4,color="red")

virus_data = df[df.isVirus == True]

plt.scatter(virus_data.feature_1,virus_data.feature_4, color="red")
plt.xlabel("feature_1")
plt.ylabel("feature_2")

df_new = df.fillna(df.mean()) #imputed nan values with mean.

df_new.hist(figsize=(20,20))

virus_data_new = df_new[df_new.isVirus == True]

plt.scatter(virus_data_new.feature_1,virus_data_new.feature_4, color="b")

sns.displot(df_new.feature_1) #skewness of the graph is descreased a little bit. 
                              #but it still has skewness

y = df_new.isVirus
x = df_new.drop(["isVirus"],axis=1)

test_size = 0.25

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size, random_state=0, shuffle = True, stratify=None)

# score = []

# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(x_train,y_train)
# score.append(classifier.score(x_test,y_test))
    
# print(score)

# Result is not good, it would be better if we clean out the outlier data.



























































