#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install liac-arff


# In[3]:


### since file is in ariff format trying to convert into csv

import arff
import csv


# In[5]:


arff_data = arff.load(open('ObesityDataset.arff','r'))


# In[6]:


data = arff_data['data']
attributes = arff_data['attributes']


# In[7]:


### converting the file to csv

with open('obesity.csv','w',newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([attr[0] for attr in attributes])
    writer.writerows(data)
    


# In[8]:


### checking if we are able to read the csv file
import pandas as pd

df = pd.read_csv('obesity.csv')
df


# In[10]:


#### importing the required libraires
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df.head()


# In[12]:


### changing the shape of the data 

print(f"data shape : {df.shape}")


# In[13]:


### checking if any null values are present in the data
df.isnull().sum()


# we can see there are no null values present inside the data

# In[14]:


### data types of the columns
df.info()


# In[15]:


df.describe()


# In[17]:


### checking if the values are correlated to each other
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(),annot = True, linewidth = .5, fmt= '.1f',ax=ax)


# In[21]:


#### removing height and weight since they are highly correlated and correlate to bmi calculation
#df = df.drop(columns=['Height','Weight'])
print(df.shape)


# In[24]:


## covert object/text to categorical variables
columns = ["Gender","family_history_with_overweight","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS","NObeyesdad"]

for column in columns:
    df[column] = df[column].astype('category')


# In[26]:


#function to interigate data after conversion
# provides min, max, unique counts
def variable_counts(columns, stage):

   if stage == 'pre':
       print("Pre Conversion to Integer")
   else:
       print("Post Conversion to Integer")

   for column in columns:    
       print("Variable:", column, "| Count Unique:",df[column].nunique(),"| Min: ", df[column].min(), "| Max: ",df[column].max())


# In[31]:


## convert float variables to nearest integer
columns = ["FCVC","NCP","CH2O","TUE","FAF"]

for column in columns:
    df[column] = round(df[column].astype('int'))

variable_counts(columns,"post")


# In[34]:


### subplots of columns
columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
           'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

fig,ax = plt.subplots(3,3, figsize=(15,10))
for column,subplot in zip(columns,ax.flatten()):
    sns.countplot(df[column], ax=subplot)

    if column == 'MTRANS':
        sns.countplot(df[column],ax=subplot)
        subplot.set_xticklabels(rotation=45,horizontalalignment='right',labels=df.MTRANS)
        subplot.yaxis.label.set_text("Number of Records")
    elif column=="NObeyesdad":
        sns.countplot(df[column],ax=subplot)
        subplot.set_xticklabels(rotation=45, horizontalalignment='right', labels=df.NObeyesdad)  
        subplot.yaxis.label.set_text("Number of Records")
    else:
        sns.countplot(df[column],ax=subplot)  
        subplot.yaxis.label.set_text("Number of Records")
        
# show figure & plots
fig.suptitle("Categorigal Variables", fontsize=20)
plt.tight_layout(pad=5, w_pad=0.0, h_pad=1)
plt.show()

    


# In[35]:


columns = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]

fig, ax = plt.subplots(1, 5, figsize=(15, 4))
for column, subplot in zip(columns, ax.flatten()):
    sns.countplot(df[column], ax=subplot)
    subplot.yaxis.label.set_text("Number of Records")

# show figure & plots
fig.suptitle("Ordinal Variables", fontsize=20)
plt.tight_layout(pad=5, w_pad=0.7, h_pad=0.5)
plt.show()


# In[36]:


fig = plt.figure(figsize = (16,5))

#distplot
ax1 = fig.add_subplot(121)
sns.distplot(df["Age"], kde=True)

#boxplot
ax1 = ax1 = fig.add_subplot(122)
sns.boxplot(df.Age)

# show figure & plots
fig.suptitle("Distribution of Numeric (Ratio) Variable", fontsize=20)
plt.tight_layout(pad=5, w_pad=0.5, h_pad=.1)
plt.show()


# In[38]:


## create an figure
fig = plt.figure(figsize=(15,5))

# add subplot for one row 2 graphs first postion
ax1 = fig.add_subplot(121)


### adding subplot for one row 2 graphs first position
matrix = np.triu(df.corr())

# set title 
ax1.title.set_text("Coorelation Heatmap: Predictor Variables")

#define plot
sns.heatmap(df.corr(), 
                 mask=matrix,
                 annot = False,                 
                 fmt='.1g', 
                 cmap="YlGnBu", 
                 vmin=-1, vmax=1, center= 0,                 
                 square="True",
                 ax=ax1)

# add second subplot
ax2 = fig.add_subplot(122)

# rotate axis label
ax2.set_xticklabels(rotation=45, horizontalalignment='right', labels=df.NObeyesdad)

# Set title text
ax2.title.set_text("Weight Category Counts: Target Variable")

# define second plot
sns.countplot(x="NObeyesdad",                  
                 palette="Blues_d", 
                 order=df.NObeyesdad.value_counts().index,
                 ax = ax2,
                 data=df)

# labels for x and y
ax2.xaxis.label.set_text("Level Category")
ax2.yaxis.label.set_text("Number of Records")

# turn off top and right frame lines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# show figure & plots
plt.tight_layout()
plt.show()


# In[39]:


### data prep for ml models
df_prep = df.copy()


# In[42]:


### create an dummy variable
df_prep = pd.get_dummies(df_prep,columns=["Gender","family_history_with_overweight",
                                          "FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"])
df_prep.head()


# In[43]:


### splitting the data into features and target variables

#Features

x = df_prep.drop(columns=['NObeyesdad'])

# Target variable

y = df_prep["NObeyesdad"]


# In[45]:


#### importing sklearn packages for data treatments

from sklearn.model_selection import train_test_split

### splitting the dataset into train and test data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)## 70% TRAIN and 30% test dataset


# using these four models might help us out to get the desired results
# 
# Decision Trees
# 
# Random Forest
# 
# Support Vector Machines (SVM)
# 
# K Nearest Neighbors

# In[56]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

### standard scale data
ss = StandardScaler()
x_train_scaled = ss.fit_transform(x_train)
x_test_scaled = ss.transform(x_test)

## tested MinMaxScaler as KNN historically does better with MinMax
mm = MinMaxScaler()
x_train_mm_scaled = ss.fit_transform(x_train)
x_test_mm_scaled = ss.transform(x_test)

### program to run multiple module through sklearn
def run_models(x_train: pd.DataFrame,y_train:pd.DataFrame,x_test: pd.DataFrame,y_test:pd.DataFrame):
    
    models = [
        ('Random Forest', RandomForestClassifier(random_state=200)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC())
    ]
    
    for name,model in models:
        # unscaled data
        clf = model.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        
        ## scaled data 
        clf_scaled = model.fit(x_train_scaled,y_train)
        y_pred_scaled = clf_scaled.predict(x_test_scaled)
        
        
        ##mm scaled data
        clf_mm_scaled = model.fit(x_train_mm_scaled, y_train)
        y_pred_mm_scaled = clf_scaled.predict(x_test_mm_scaled)
        
        # accuracy scores
        accuracy = round(metrics.accuracy_score(y_test, y_pred),5)
        scaled_accuracy = round(metrics.accuracy_score(y_test, y_pred_scaled),5)
        scaled_mm_accuracy = round(metrics.accuracy_score(y_test, y_pred_mm_scaled),5)
        
        # output
        print(name + ':')        
        print("---------------------------------------------------------------")      
        print("Accuracy:", accuracy)
        print("Accuracy w/Scaled Data (ss):", scaled_accuracy)
        print("Accuracy w/Scaled Data (mm):", scaled_mm_accuracy)
        if (accuracy > scaled_accuracy) and (accuracy > scaled_mm_accuracy):
            print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))      
            print("                            -----------------------------------               \n")      
        elif (scaled_accuracy > scaled_mm_accuracy):
            print("\nClassification Report (ss):\n", metrics.classification_report(y_test, y_pred_scaled))      
            print("                            -----------------------------------               \n")     
        else:            
            print("\nClassification Report (mm):\n", metrics.classification_report(y_test, y_pred_mm_scaled))      
            print("                            -----------------------------------               \n")      


# In[57]:


#run Decision Trees, Random Forest, KNN and SVM
run_models(x_train, y_train, x_test, y_test)


# # Hyperparameter Tuning

# In[58]:


from sklearn.model_selection import GridSearchCV

#model name, classifier, parameters
# function used to process models and parameters through gridsearch
def hyper_tune(name, clf, parameters, target_names=None): 
    
    target_names = target_names
    clf = clf
    search = GridSearchCV(clf, parameters,verbose=True, n_jobs=15, cv=5)
    search.fit(x_train_scaled,y_train)
    y_pred_scaled = search.predict(x_test_scaled)
    print ("Accuracy Score = %3.2f" %(search.score(x_test_scaled,y_test)))
    print (search.best_params_)
    print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred_scaled, target_names=target_names))
    


# In[59]:


#the KNN model performs better on the unscaled data this function
# function for unscaled data
#model name, classifier, parameters
# function used to process models and parameters through gridsearch
def hyper_tune2(name, clf, parameters, target_names=None): 
    
    target_names = target_names
    clf = clf
    search = GridSearchCV(clf, parameters,verbose=True, n_jobs=15, cv=5)
    search.fit(x_train,y_train)
    y_pred = search.predict(x_test)
    print ("Accuracy Score = %3.2f" %(search.score(x_test,y_test)))
    print (search.best_params_)
    print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred, target_names=target_names))


# # KNN

# In[60]:


# Number of neighbors
n_neighbors = [int(x) for x in range(4, 15)]
# weights
weights = ['uniform','distance']
# distance metric
metric = ['euclidean', 'manhattan', 'chebyshev']
# computation algorithm
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
# power paramter
p=[1,2]

parameters = { 'n_neighbors': n_neighbors,
              'weights':weights,
              'metric':metric,
              'p':p,
              'algorithm': algorithm              
               }

hyper_tune2('KNN', KNeighborsClassifier(), parameters)


# # Random Forest

# In[61]:


# Number of trees in random forest
n_estimators = [int(x) for x in range(10, 200,10)]
# Criterion
criterion = ['gini','entropy']
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(10, 100, 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in range(2, 5)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in range(2, 5)]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# random state
random_state = [1010]

parameters = { 'criterion':criterion,
               'n_estimators': n_estimators,
              'max_depth':max_depth,
              #'random_state': random_state,
              #'max_features':max_features,
              #'min_samples_split':min_samples_split             
               }


hyper_tune('Random Forest',
           RandomForestClassifier(), parameters)


# # Feature Importance w/ Random Forest

# In[62]:


# Create Decision Tree classifer object with optimized parameters
clf = RandomForestClassifier(criterion='entropy',
               n_estimators=52,
              max_depth = 51,              
              max_features='auto',
              min_samples_split=2,
              random_state=1010)

# Train Decision Tree Classifer
clf = clf.fit(x_train_scaled,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test_scaled)


# In[63]:


print(x.columns)


# In[66]:


feature_imp = pd.Series(clf.feature_importances_, index=x.columns).sort_values(ascending=False)
fig = plt.figure(figsize=(10, 5))

# Creating a bar plot
sns.barplot(x=feature_imp.index, y=feature_imp)

# Add labels to your graph
plt.xticks(rotation=45, horizontalalignment='right')

plt.tight_layout()
plt.show()

# create features list
features_list = x.columns
features_list = features_list.tolist()

# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
print("\nTop 10 Features:")
display_top = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:10]]

# Sort the feature importances by least important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)
# Print out the feature and importances 
print("\nBottom 10 Features:")
display_bottom = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:10]]


# # Data Prep
# 
# Underweight mapped to being not obese. Arguably however it is just as much of a health concern as being overweight.

# In[67]:


# map values 
weight_map = { 'Normal_Weight':0, 'Overweight_Level_I':0,
               'Overweight_Level_II':0, 'Obesity_Type_I':1,
               'Obesity_Type_II':1, 'Obesity_Type_III':1, 'Insufficient_Weight':0}

# map values
df_prep['weight_cat'] = df_prep['NObeyesdad'].map(weight_map)


# In[68]:


sns.countplot(x="weight_cat",                  
                 palette="Blues_d", 
                 order=df_prep["weight_cat"].value_counts().index,                 
                 data=df_prep)


# show figure & plots
plt.tight_layout()
plt.show()


# In[69]:


# split dataset in features and target variable

# Features
X = df_prep.drop(columns=["NObeyesdad","weight_cat"])

# Target variable
y = df_prep['weight_cat'] 


# In[70]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Scaled version of X train and X test
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


# # Feature Importance with Random Forest: Two Category Variable Target

# In[71]:


# Create Random Forest classifer object with optimized parameters
clf = RandomForestClassifier(criterion='gini',
               n_estimators=110,
              max_depth = 20,              
              max_features='auto',              
              random_state=1010)

# Train Random Forest classifer
clf = clf.fit(X_train_scaled,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test_scaled)


# In[72]:


feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig = plt.figure(figsize=(12, 5))

# Creating a bar plot
sns.barplot(x=feature_imp.index, y=feature_imp)

# Add labels to your graph
plt.xticks(rotation=45, horizontalalignment='right')

plt.tight_layout()
plt.show()

# create features list
features_list = X.columns
features_list = features_list.tolist()

# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
print("\nTop 10 Features:")
display_top = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:10]]

# Sort the feature importances by least important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = False)
# Print out the feature and importances 
print("\nBottom 10 Features:")
display_bottom = [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:10]]


# In[ ]:




