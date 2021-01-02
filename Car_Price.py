#!/usr/bin/env python
# coding: utf-8

# # importing libraries and data and basic analysis of the data

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


data = pd.read_csv("CarPrice_Assignment.csv")
data


# In[10]:


data.shape


# In[5]:


data.describe


# In[6]:


data.describe()


# In[8]:


data.info()


# # Step 2: Data cleaning and preparation

# In[11]:


#splitting company name from CarName column
CompanyName = data['CarName'].apply(lambda x:x.split(' ')[0])
data.insert(3, "CompanyName", CompanyName)#insert the column in 4th column
data.drop(['CarName'], axis = 1, inplace = True)#drop CarName column
data.head()


# In[12]:


#to return all the unique values in column CompanyName
#some spelling errors are detected in the output for example maxda and mazda, Nissan and nissan, porsche and porcshce, etc.
data.CompanyName.unique()


# In[13]:


#correcting the above spelling errors using replace function
data.CompanyName = data.CompanyName.str.lower()#converting all into lowercase

def replace_name(a, b):
    data.CompanyName.replace(a, b, inplace=True)

    
replace_name('maxda', 'mazda')
replace_name('porcshce', 'porsche')
replace_name('toyouta', 'toyota')
replace_name('vokswagen', 'volkswagen')
replace_name('vw', 'volkswagen')

data.CompanyName.unique()


# In[14]:


#checking duplicates
data.loc[data.duplicated()]
#no duplicates found


# In[8]:


data.columns


# # Step 3:Visualising the data

# In[19]:


plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.title('Car Price Distribution Plot')
#used for univariate analysis using a histogram using a particular column
sns.distplot(data.price)

plt.subplot(1, 2, 2)
plt.title('Car Price Spread')
#boxplot is used for quartiles and also to find outliers in the data. 
sns.boxplot(y = data.price)
#most of the prices are below 15000 so plot seems to be right skewed

plt.show()


# In[22]:


data.price.describe(percentiles = [0.25, 0.50, 0.75, 0.85, 0.90, 1])
#there is significant difference between mean and median
#85% of prices are less than 18500 and rest 15% lie between 18,500 ad 45,400. It indicates high variance in the car prices


# #Visualising Categorical Data( CompanyName, Symboling, fueltype, enginetype, carbody, doornumber, enginelocation, fuelsystem, cylindernumber, aspiration, drivewheel)

# In[22]:


#Analysis of CompanyName, fueltype, carbody
plt.figure(figsize = (25, 6))

plt.subplot(1, 3, 1)
plt1 = data.CompanyName.value_counts().plot('bar')
#value_counts() returns count of unique values in descending order
plt.title('Companies Histogram')
plt1.set(xlabel='Car company', ylabel = 'Frequency of company')

plt.subplot(1, 3, 2)
plt1 = data.fueltype.value_counts().plot('bar')
plt.title('Fuel Type Histogram')
plt1.set(xlabel='Fuel Type', ylabel = 'Frequency of fuel type')

plt.subplot(1, 3, 3)
plt1 = data.carbody.value_counts().plot('bar')
plt.title('Car Type Histogram')
plt1.set(xlabel='Car Type', ylabel = 'Frequency of Car type')

plt.show()
#from the bar graphs we can conclude that toyota is the most favoured company with frequency greater than 30. 
#gas fueled cars have more frequency than the diesel ones.(ask Krishnaai how to calculate percentage)
#sedan is most prefered while convertibles are least prefered car types.


# In[24]:


#Analysis of symboling
plt.figure(figsize = (20,8))

plt.subplot(1, 2, 1)
plt.title('Symboling Histogram')
sns.countplot(data.symboling, palette = ("cubehelix"))#shows value count for a single categorical variable, palette is used for color
#cubehelix is a type of color palette
plt.subplot(1, 2, 2)
plt.title('Symboling VS Price')
sns.boxplot(x = data.symboling, y = data.price,  palette = ("cubehelix"))

plt.show()
#cars are assigned a risk factor symbol associated with its price known as symboling.+3 value indicates auto is quite risky and -3 indicates it is pretty safe
#in graph 1 symboling with 0 and 1 values have most frequency i.e. they are most sold and that with -2 are least sold.
#in graph 2 cars with symboling -1 are highly priced and -2 also. But cars with symboling 3 are also highly priced.
#Cars with symboling value 1 are with lowest price.


# In[27]:


#Analysis of enginetype
plt.figure(figsize = (15,8))

plt.subplot(1, 2, 1)
plt.title('Engine Type Histogram')
sns.countplot(data.enginetype, palette = ("rocket_r"))#shows value count for a single categorical variable, palette is used for color

plt.subplot(1, 2, 2)
plt.title('Engine Type VS Price')
sns.boxplot(x = data.enginetype, y = data.price,  palette = ("viridis_r"))

plt.show()
#We can conclude that ohc is most used Engine Type. dohcv is least used.
#ohcv engine has maximum price, ohc and ohcf have low price range(WHile dohcv has only one row)


# In[30]:


plt.figure(figsize = (25,6))

df = pd.DataFrame(data.groupby(['enginetype'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('Engine Type VS Average Price')
plt.show()

df = pd.DataFrame(data.groupby(['CompanyName'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('Company Name VS Average Price')
plt.show()

df = pd.DataFrame(data.groupby(['fueltype'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('Fuel Type VS Average Price')
plt.show()

df = pd.DataFrame(data.groupby(['carbody'])['price'].mean().sort_values(ascending=False))
df.plot.bar()
plt.title('Car Type VS Average Price')
plt.show()

#dohcv engine has highest average price
#Jaguar and Buick has highest average price
#diesel has more average price than gas fuel type
#hardtop and convertible type cars have more average price


# In[38]:


#Analysis of doornumber 
plt.figure(figsize = (15,5))

plt.subplot(1, 2, 1)
plt.title('Door Number Histogram')
sns.countplot(data.doornumber, palette = ("plasma"))

plt.subplot(1, 2, 2)
plt.title('Door Number VS Price')
sns.boxplot(x = data.doornumber, y = data.price,  palette = ("plasma"))

plt.show()

#Analysis of aspiration(it is an internal combustion system)

plt.figure(figsize = (15,5))

plt.subplot(1, 2, 1)
plt.title('Aspiration Histogram')
sns.countplot(data.aspiration, palette = ("Paired_r"))

plt.subplot(1, 2, 2)
plt.title('Aspiration VS Price')
sns.boxplot(x = data.aspiration, y = data.price,  palette = ("Paired_r"))

plt.show()

#Door Number is not affecting the car price much as their is no significant difference between the categories. 
#Aspiration with std have more count compared to turbo. Turbo has more price than std though their are many outsiders in std.


# In[39]:


#Analysis of features enginelocation, cylindernumber, fuelsystem, drivewheel
def plot_count(x, fig):
    plt.subplot(4, 2, fig)
    plt.title(x+' Histogram')
    sns.countplot(data[x], palette=("magma"))
    plt.subplot(4, 2, (fig+1))
    plt.title(x+' VS Price')
    sns.boxplot(x=data[x], y = data.price, palette=("magma"))
    
    
plt.figure(figsize = (15, 20))

plot_count('enginelocation', 1)
plot_count('cylindernumber', 3)
plot_count('fuelsystem', 5)
plot_count('drivewheel', 7)

plt.tight_layout()

#most common number of cylinders are four, six and five, but eight cylinder has maximum price 
#mpfi and 2bbl are most common type of fuel systems. mpfi and idi have highest price
#fwd drivewheel is most preffered whereas rwd drivewheel has maximum price range


# Visualising Numerical Data 

# In[3]:


#Analysis of features carlength, carwidth, carheight, curbweight
def scatter(x, fig):
    plt.subplot(5, 2, fig)
    plt.scatter(data[x], data['price'])#scatter plot used to visualise the data
    plt.title(x+' VS Price')
    plt.ylabel('Price')
    plt.xlabel(x)
    
plt.figure(figsize = (10, 20))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()

#carlength, carwidth and curbweight seem to have a positive correlation with price
#carheight does'nt show any significant trend with price


# In[4]:


def pp(x, y, z):
    sns.pairplot(data, x_vars=[x, y, z], y_vars='price', size = 4, aspect = 1, kind = 'scatter')
    plt.show()

pp('enginesize', 'boreratio', 'stroke' )
pp('compressionratio', 'horsepower', 'peakrpm' )
pp('wheelbase', 'citympg', 'highwaympg' )

#enginesize, boreratio, horsepower, wheelbase seem to have positive correlation with price
#citympg, highwaympg seem to have negative correlation with price


# In[5]:


np.corrcoef(data['carlength'], data['carwidth'])[0,1]


# In[9]:


np.corrcoef(data['enginesize'], data['price'])[0,1]


# In[10]:


np.corrcoef(data['highwaympg'], data['price'])[0,1]


# In[8]:


np.corrcoef(data['citympg'], data['price'])[0,1]


# # Step 4 : Deriving new features

# In[15]:


#Fuel Economy
data['fueleconomy'] = (0.55*data['citympg']) + (0.45*data['highwaympg'])


# In[16]:


#Binning the Car Companies based on avg prices of each Company
data['price']=data['price'].astype('int')
temp = data.copy()
table = temp.groupby(['CompanyName'])['price'].mean()
temp = temp.merge(table.reset_index(), how='left', on = 'CompanyName')
bins = [0,10000, 20000, 40000]
cars_bin=['Budget', 'Medium', 'Highend']
data['carsrange'] = pd.cut(temp['price_y'], bins, right=False, labels=cars_bin)
data.head()


# # Step 5 : Bivariate Analysis :

# In[17]:


plt.figure(figsize = (8, 6))

plt.title('Fuel economy VS Price')
sns.scatterplot(x = data['fueleconomy'], y = data['price'], hue = data['drivewheel'])
plt.xlabel('Fuel Economy')
plt.ylabel('Price')

plt.show()
plt.tight_layout()
#fueleconomy has a negative correlation with price


# In[14]:


plt.figure(figsize = (25, 6))


df = pd.DataFrame(data.groupby(['fuelsystem', 'drivewheel', 'carsrange'])['price'].mean().unstack(fill_value=0))
df.plot.bar()
plt.title('Fuel economy VS Price')

plt.show()

#We can conclude that high ranged cars prefer rwd drivewheel with idi or mpfi fuelsystem


# List of significant variables after Visual Analysis:
# 1. Car Range
# 2. Engine Type
# 3. Fuel type
# 4. Car Body
# 5. Aspiration
# 6. Cylinder Number
# 7. Drivewheel
# 8. Curbweight
# 9. Car Length
# 10. Car width 
# 11. Engine Size
# 12. Boreration
# 13. Horse Power
# 14. Wheel base
# 15. Fuel Economy

# In[17]:


cars_lr = data[['price','fueltype', 'aspiration', 'carbody','drivewheel','wheelbase','curbweight','enginetype','cylindernumber','enginesize','boreratio','horsepower', 'fueleconomy', 'carlength', 'carwidth', 'carsrange']]
cars_lr.head()


# In[17]:


sns.pairplot(cars_lr)
plt.show()


# # Step 6 : Dummy Variables

# In[18]:


#defining the map function
def dummies(x, df):
    temp = pd.get_dummies(df[x], drop_first = True)#converts categorical data into dummy or indicator variables.
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

#Applying the function to cars_lr

cars_lr = dummies('fueltype', cars_lr)
cars_lr = dummies('aspiration', cars_lr)
cars_lr = dummies('carbody', cars_lr)
cars_lr = dummies('drivewheel', cars_lr)
cars_lr = dummies('enginetype', cars_lr)
cars_lr = dummies('cylindernumber', cars_lr)
cars_lr = dummies('carsrange', cars_lr)


# In[19]:


cars_lr.head()


# In[20]:


cars_lr.shape


# # Step 7 : Train-Test Split and Feature Scalling

# In[21]:


from sklearn.model_selection import train_test_split
np.random.seed(0) #np.random.seed(0) makes the random numbers predictable
df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size=0.3, random_state=100)


# In[22]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'fueleconomy', 'carlength', 'carwidth', 'price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[16]:


df_train.head()


# In[17]:


df_train.describe()


# In[25]:


#Correlation using heatmap
plt.figure(figsize = (30, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="BuGn")
plt.show()

#from the figure below we can conclude that curbweight, enginesize, horsepower, carwidth and highend are highly correlated with price 


# In[23]:


#dividing data into X and y variables
y_train = df_train.pop('price')
X_train = df_train


# # Step 8 : Model Building

# In[24]:


#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
#import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[25]:


lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)


# In[52]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[26]:


X_train.columns[rfe.support_]


# Building Model using statsmodel, for the detailed statistics

# In[27]:


X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()


# In[35]:


def build_model(X, y):
    X = sm.add_constant(X) #adding the constant
    lm = sm.OLS(y, X).fit() #fitting the model
    print(lm.summary()) #model summary
    return X

def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)


# #MODEL 1

# In[29]:


X_train_new = build_model(X_train_rfe, y_train)


# In[30]:


X_train_new = X_train_rfe.drop(["twelve"], axis = 1)#p-vale of twelve seems to be higher than significance value of 0.05, thus dropping it as it is insignificant in presence of other variables


# MODEL 2

# In[31]:


X_train_new = build_model(X_train_new, y_train)


# In[32]:


X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)


# MODEL 3

# In[33]:


X_train_new = build_model(X_train_new,y_train)


# In[36]:


#Calculating the Variance Inflation Factor
checkVIF(X_train_new)


# In[37]:


#we will drop curbweight since it has hight VIF(it means that curbweight has high multicolinearity)

X_train_new = X_train_new.drop(["curbweight"], axis = 1)


# MODEL 4

# In[38]:


X_train_new = build_model(X_train_new,y_train)


# In[39]:


checkVIF(X_train_new)


# In[40]:


#dropping sedan because of high VIF value
X_train_new = X_train_new.drop(["sedan"], axis = 1)


# MODEL 5

# In[41]:


X_train_new = build_model(X_train_new,y_train)


# In[42]:


checkVIF(X_train_new)


# In[43]:


#dropping wagon because of high p value
X_train_new = X_train_new.drop(["wagon"],axis = 1)


# MODEL 6

# In[44]:


X_train_new = build_model(X_train_new,y_train)


# In[45]:


checkVIF(X_train_new)


# MODEL 7

# In[46]:


#Dropping dohcv to see the changes in model statistics
X_train_new = X_train_new.drop(["dohcv"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)


# # Step 9 : Residual Analysis of Model

# In[47]:


lm = sm.OLS(y_train, X_train_new).fit()
y_train_price = lm.predict(X_train_new)


# In[48]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)# Plot heading 
plt.xlabel('Errors', fontsize = 18) 
#Error terms seem to be approximately normally distributed, so the assumption on the linear modeling seems to be fulfilled.


# # Step 10 : Prediction and Evaluation

# In[49]:


#Scaling the test set
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[50]:


#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test


# In[51]:


# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]
# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[52]:


# Making predictions
y_pred = lm.predict(X_test_new)


# Evaluation of test via comparison of y_pred and y_test

# In[53]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[54]:


#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)# Plot heading 
plt.xlabel('y_test', fontsize=18)# X-label
plt.ylabel('y_pred', fontsize=16)   


# Evaluation of the model using Statistics

# In[55]:


print(lm.summary())


# In[56]:


#CONCLUSIONS :

#R-sqaured and Adjusted R-squared (extent of fit) - 0.899 and 0.896 - 90% variance explained.
#F-stats and Prob(F-stats) (overall model fit) - 308.0 and 1.04e-67(approx. 0.0) - Model fir is significant and explained 90% variance is just not by chance.
#p-values - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the predictors are statistically significant.

