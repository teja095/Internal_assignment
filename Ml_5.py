#!/usr/bin/env python
# coding: utf-8

# In[1]:


## importing different libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


### reading the dataset
df = pd.read_csv('rideshare.csv')
df.head(10)


# In[5]:


### checking the data size 
df.shape


# In[6]:


### checoking the datatype of dataset
df.info()


# In[7]:


### checking if there is any missing values in the dataset
df.isnull().sum()


# there is no null values except for the price which can be ignored accordingly with respect to the further calculations

# In[8]:


## description of the dataset
df.describe()


# # there are total 46 columns we have taken only few for analyizing the price of it

# In[9]:


columns = ["id","month","day","hour","distance","surge_multiplier","cab_type","latitude","longitude",
           "temperature","precipIntensity","precipProbability","humidity","windSpeed","windGust","dewPoint",
           "pressure","ozone","price"]
analysis_df = df[columns]


# In[10]:


analysis_df.head()


# In[15]:


### visualize the relationship between the features and response using scatterplots
fig,axs = plt.subplots(1,2,sharey=True)
df.plot(kind='scatter',x='distance',y='price',ax=axs[0],figsize=(16,8))

df.plot(kind='scatter',x='hour',y='price',ax=axs[1])


# In[17]:


df = analysis_df.groupby(by=['month']).size().reset_index(name='counts')
df


# In[18]:


import plotly.express as px


# In[19]:


px.bar(data_frame=df,x='month',y='counts',color='month',barmode='group')


# In[20]:


df =analysis_df.groupby(by=["day"]).size().reset_index(name="counts")
df


# In[21]:


px.bar(data_frame=df, x="day", y="counts", color="day", barmode="group")


# we can see almost for 15 days there no booking we need to find why it is so, but we have more number of bookings in the month of december

# In[22]:


df =analysis_df.groupby(by=["hour"]).size().reset_index(name="counts")
px.bar(data_frame=df, x="hour", y="counts", color="hour", barmode="group")


# In[23]:


df =analysis_df.groupby(by=["month","cab_type"]).size().reset_index(name="counts")
px.bar(data_frame=df, x="month", y="counts", color="cab_type", barmode="group")


# In[24]:


df =analysis_df.groupby(by=["hour","cab_type"]).size().reset_index(name="counts")
px.bar(data_frame=df, x="hour", y="counts", color="cab_type", barmode="group")


# As comparison uber has edge over lyft for all accessed timzone 

# In[26]:


## checking the correlation
f = plt.figure(figsize=(19, 15))
plt.matshow(analysis_df.corr(), fignum=f.number)
plt.xticks(range(analysis_df.select_dtypes(['number']).shape[1]), analysis_df.select_dtypes(['number']).columns, fontsize=12, rotation=45)
plt.yticks(range(analysis_df.select_dtypes(['number']).shape[1]), analysis_df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# In[28]:


import seaborn as sns
sns.set(style = "darkgrid")
surge_data = analysis_df[analysis_df["surge_multiplier"]>2.0]
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection = '3d')

x = surge_data['distance']
y = surge_data['price']
z = surge_data['surge_multiplier']

ax.set_xlabel("distance")
ax.set_ylabel("price")
ax.set_zlabel("surge_multiplier")

ax.scatter(x, y, z)

plt.show()


# In[29]:


df = analysis_df[(analysis_df["hour"]>=22) | (analysis_df["hour"]<=4)]
plt.figure(figsize=(16,9))
sns.scatterplot(data = df, x = "distance",
                y = "price", hue = "cab_type", size = "surge_multiplier")
plt.show()


# In[30]:


sns.scatterplot(data = surge_data, x = "distance",
                y = "price", hue = "cab_type", size = "surge_multiplier")
plt.show()


# In[61]:


df1 = analysis_df[["distance","surge_multiplier",
           "price"]]
df1.dropna(inplace=False)


# In[62]:


# importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[63]:


#separate the other attributes from the predicting attribute
x = df1.drop("price",axis=1)
#separte the predicting attribute into Y for model training 
y = df1["price"]


# In[64]:


# importing module
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(x_train,y_train)


# In[65]:


y_prediction =  LR.predict(x_test)


# In[66]:


# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
score=r2_score(y_test,y_prediction)
print('r2 socre is ',score)
print('mean_sqrd_error is==',mean_squared_error(y_test,y_prediction))
print('root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction)))


# In[67]:


import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[69]:


LR.score(x_train,y_train)


# In[79]:


def adj_r2(x,y):
    r2 = LR.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[75]:


from sklearn.preprocessing import StandardScaler 
scaler =StandardScaler()

X_scaled = scaler.fit_transform(x)


# In[76]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_scaled

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
# we do not include categorical values for mulitcollinearity as they do not provide much information as numerical ones do
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = x.columns


# In[77]:


vif


# In[80]:


adj_r2(x_train,y_train)


# In[82]:


# Lasso Regularization
# LassoCV will return best alpha and coefficients after performing 10 cross validations
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
lasscv.fit(x_train, y_train)


# In[83]:


# best alpha parameter
alpha = lasscv.alpha_
alpha


# In[84]:


#now that we have best parameter, let's use Lasso regression and see how well our data has fitted before

lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train, y_train)


# In[85]:


lasso_reg.score(x_test, y_test)


# In[86]:


# Using Ridge regression model
# RidgeCV will return best alpha and coefficients after performing 10 cross validations. 
# We will pass an array of random numbers for ridgeCV to select best alpha from them

alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(x_train, y_train)


# In[87]:


ridgecv.alpha_


# In[88]:


ridge_model = Ridge(alpha=ridgecv.alpha_)
ridge_model.fit(x_train, y_train)


# In[89]:


ridge_model.score(x_test, y_test)


# In[90]:


# Elastic net

elasticCV = ElasticNetCV(alphas = None, cv =10)

elasticCV.fit(x_train, y_train)


# In[91]:


elasticCV.alpha_


# In[92]:


# l1_ration gives how close the model is to L1 regularization, below value indicates we are giving equal
#preference to L1 and L2
elasticCV.l1_ratio


# In[93]:


elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)
elasticnet_reg.fit(x_train, y_train)


# In[94]:


elasticnet_reg.score(x_test, y_test)


# now we are going to use unsupervised algorithm for high booking area

# In[138]:


from sklearn.cluster import KMeans


# In[137]:


df2=df.copy()


# fitting for all data except destination

# In[146]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2['destination']= le.fit_transform(df2['destination'])
df2['destination'].value_counts()


# In[139]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2['source']= le.fit_transform(df2['source'])
df2['source'].value_counts()


# In[140]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df2['name']= le.fit_transform(df2['name'])
df2['name'].value_counts()


# In[129]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['cab_type']= le.fit_transform(df1['source'])
df1['cab_type'].value_counts()


# In[141]:


df2.drop(['timezone','id'],axis=1,inplace=True)


# In[142]:


df2.drop(['datetime','product_id','short_summary','long_summary','icon'],axis=1,inplace=True)


# In[143]:


df2.dropna(inplace=True)


# In[119]:


from sklearn.cluster import KMeans


# In[147]:


kmeans = KMeans(n_clusters=2)


# In[148]:


kmeans.fit(df1.drop('destination',axis=1))


# In[149]:


kmeans.cluster_centers_


# In[150]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[153]:


df2['Cluster'] = df2['cab_type'].apply(converter)


# In[154]:


df.head()


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# 
# [[138  74]
#  
#  [531  34]]
#               
#               precision    recall  f1-score   support
# 
#            0       0.21      0.56      0.31       212
#            1       0.31      0.03      0.10       565
# 
#     accuracy                           0.22       777
#     macro avg       0.21      0.36     0.21       777
#    
#     weighted avg    0.28      0.22     0.16       777

# In[ ]:




