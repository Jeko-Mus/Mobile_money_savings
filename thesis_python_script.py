#!/usr/bin/env python
# coding: utf-8

# # Data analysis of the Zimbabwe global findex database 2017
# ##### by Rujeko Musarurwa
# This database is a large dataset that contains 1000 individuals information with over 100 different variables of which I have chosen to focus on 10 relevant variables for my analysis. The aim of this exploration is to assess the data and any relationships that may exist between the different attributes with the main variable of interest being savings.
# The main feature in the dataset is the influence on savings by having a mobile money account.
# 
# Logistic Regression should be used because the response variable is categorical that can only predict two possible outcomes of saved or did not save.

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load Zimbabwe dataset into dataframe called df
df = pd.read_excel('thesis_dataset_zim_data.xls')
df.shape


# In[3]:


#check if data loaded correctly
df.head()


# In[4]:


#replace the spaces between column names with underscore
df.columns = df.columns.str.replace(' ', '_')


# In[5]:


#check above has been done
df.head()


# In[6]:


#rename(shorten) some variable names to make it easier for anaylsis
df.rename(columns={"Gallup_World_Poll_identifier":"unique_id", "Respondent_is_female":"female", "Respondent_age":"age", 
                   "Respondent_education_level":"education", "Within-economy_household_income_quintile":"hh_income_quintile",
                   "Owns_a_mobile_phone":"owns_mobile_phone", "Respondent_is_in_the_workforce":"in_workforce",
                   "Has_a_mobile_money_account":"mobile_money_account", "Saved_in_the_past_year":"saved", 
                   "Saved_in_past_12_months:_for_farm/business_purposes":"saved_farm_business",
                  "Saved_in_past_12_months:_for_old_age":"saved_old_age"}, inplace=True)


# In[7]:


# check the datatypes are correct
df.info()


# In[8]:


#check unique variables
df.nunique()


# ## check for missing values

# In[9]:


# checking columns that have missing data
df.isnull().sum()


# In[10]:


# replace values written as dk(don't know), rf(refused) or ref as missing values
df.replace({'(dk)':np.nan,'(rf)':np.nan,'(ref)':np.nan}, inplace=True)

#drop missing values
df.dropna(inplace=True)

#reset the index
df.reset_index(drop=True, inplace=True)


# In[11]:


#check the values have been removed
df.info()


# In[12]:


#check all variables that should have two outcomes have two outcomes
df.nunique()


# In[13]:


df.head()


# ### replacing certain string values with others

# In[14]:


#replace all 0 values for last four columns with a 'no' so that we have yes or no instead of yes and zero.
df["mobile_money_account"].replace({str(0): 'no'}, inplace=True)
df["saved"].replace({str(0): 'no'}, inplace=True)


# In[15]:


#check corrections have been done
df.head()


# In[16]:


#check all variables have two outcomes except education and income quintile
df.nunique()


# ## Univariate exploration of variables

# In[17]:


#number of females and males
df.female.value_counts()


# In[18]:


#number of respondents that save and not save
df.saved.value_counts()


# In[19]:


df.saved_farm_business.value_counts()


# In[20]:


df.saved_old_age.value_counts()


# In[21]:


df.owns_mobile_phone.value_counts()


# In[22]:


df.mobile_money_account.value_counts()


# In[23]:


df.in_workforce.value_counts()


# In[24]:


df.education.value_counts()


# In[25]:


df.education.value_counts().plot(kind='pie');


# In[26]:


df.hh_income_quintile.value_counts()


# In[27]:


plt.figure(figsize = [12, 4])
#5 categories - poorest,second,middle,fourth,richest. i.e poorest, second poorest etc...
cat_order = df['hh_income_quintile'].value_counts().index
sb.countplot(data = df, x = 'hh_income_quintile', order = cat_order);


# In[28]:


df.age.describe()


# In[29]:


plt.hist(data=df, x= 'age', bins=30, label='ages')
plt.legend();


# ## Bivariate exploration of variables

# In[30]:


# use df2 specifically only to see correlations of categorical values before changing them into dummy variables
df2=df.replace({'yes':1, 'no': 0})


# In[31]:


# heatmap showing correlations
sb.heatmap(df2.corr(), annot = True, fmt = '.3f',
           cmap = 'vlag_r', center = 0)
plt.show()


# In[32]:


saved_vs_mobile_account = pd.crosstab(index=df["saved"], 
                          columns=df["mobile_money_account"])

saved_vs_mobile_account


# In[33]:


saved_fb_vs_mobile_account = pd.crosstab(index=df["saved_farm_business"], 
                          columns=df["mobile_money_account"])

saved_fb_vs_mobile_account


# In[34]:


saved_age_vs_mobile_account = pd.crosstab(index=df["saved_old_age"], 
                          columns=df["mobile_money_account"])

saved_age_vs_mobile_account


# In[35]:


saved_and_owns_mobile = pd.crosstab(index=df["mobile_money_account"], 
                          columns=df["owns_mobile_phone"])

saved_and_owns_mobile


# In[36]:


pd.crosstab([df.saved], df.education)


# In[37]:


pd.crosstab([df.saved_old_age], df.education)


# In[38]:


pd.crosstab([df.saved_farm_business], df.education)


# In[39]:


pd.crosstab([df.saved], df.hh_income_quintile)


# In[40]:


pd.crosstab([df.saved_old_age], df.hh_income_quintile)


# In[41]:


pd.crosstab([df.saved_farm_business], df.hh_income_quintile)


# # regression
# ### Logistic Regression because the response variable is categorical that can only predict two possible outcomes.

# In[42]:


#check to see which variables need to be prepared for the regression
df.head(2)


# In[43]:


#create age squared variable
mean_age=df.age.mean() # find average age
age_cen=df.age-mean_age # subtract average age from each age value
df['age_sqr']=age_cen*age_cen # create age squared variable


# In[44]:


#replace all yes values with 1 and all no's with 0, 
#for sex, female=1 and male=0
#for workforce, in_workforce=1 and out of workforce=0
df.replace({'yes':1, 'no': 0}, inplace=True)
df.replace({'Female':1, 'Male': 0}, inplace=True)
df.in_workforce.replace({'in workforce':1, 'out of workforce': 0}, inplace=True)
df.head()


# In[45]:


#For education and hh_income create dummy variables for them for the regression
education =pd.get_dummies(df['education'], drop_first=True)
income=pd.get_dummies(df['hh_income_quintile'], drop_first=True)


# In[46]:


# add the dummy variobles to the dataframe
df = pd.concat([df,education,income], axis=1)


# In[47]:


#check its done
df.head(1)


# In[48]:


#drop extra variables not being used anymore since we now have dummy variables for them
df.drop(['unique_id','education','hh_income_quintile','owns_mobile_phone'],axis=1,inplace=True)


# In[49]:


#check
df.head()


# ## Run different regressions

# In[50]:


#import package used for regression
import statsmodels.api as sm


# In[51]:


# the logit regression model for saved
df['intercept']=1 #creating the intercept
logit_model = sm.Logit(df['saved'],df[['intercept', 'mobile_money_account']])
results1=logit_model.fit()
results1.summary()


# In[52]:


#calculating odds ratios
print(np.exp(results1.params))


# In[53]:


# the logit regression model for saved old age
df['intercept']=1 #creating the intercept
logit_model = sm.Logit(df['saved_old_age'],df[['intercept', 'mobile_money_account']])
results2=logit_model.fit()
results2.summary()


# In[54]:


#calculating odds ratios
print(np.exp(results2.params))


# In[55]:


# the logit regression model for saved farm business
df['intercept']=1 #creating the intercept
logit_model = sm.Logit(df['saved_farm_business'],df[['intercept', 'mobile_money_account']])
results3=logit_model.fit()
results3.summary()


# In[56]:


print(np.exp(results3.params))


# In[57]:


#include controls
logit_model = sm.Logit(df['saved'],df[['intercept', 'mobile_money_account','female','age', 'age_sqr','in_workforce',
                                       'completed tertiary or more','secondary', 
                                       'Poorest 20%','Second 20%','Middle 20%', 'Richest 20%']])
results4=logit_model.fit()
results4.summary()


# In[58]:


print(np.exp(results4.params))


# In[59]:


logit_model = sm.Logit(df['saved_old_age'],df[['intercept', 'mobile_money_account','female','age', 'age_sqr','in_workforce',
                                       'completed tertiary or more','secondary', 
                                       'Poorest 20%','Second 20%','Middle 20%', 'Richest 20%']])
results5=logit_model.fit()
results5.summary()


# In[60]:


print(np.exp(results5.params))


# In[61]:


logit_model = sm.Logit(df['saved_farm_business'],df[['intercept', 'mobile_money_account','female','age', 'age_sqr',
                                       'in_workforce','completed tertiary or more','secondary', 
                                       'Poorest 20%','Second 20%','Middle 20%', 'Richest 20%']])
results6=logit_model.fit()
results6.summary()


# In[62]:


print(np.exp(results6.params))


# ## Make regression outputs look more presentable

# In[63]:


#import library required for regression summary tables
from statsmodels.iolib.summary2 import summary_col


# In[64]:


#summary table for regression for saved variable with controls
res1=summary_col([results4],stars=True,float_format='%0.2f',
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))})
print(res1)


# In[65]:


res2=summary_col([results5],stars=True,float_format='%0.2f',
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))})
print(res2)


# In[66]:


res3=summary_col([results6],stars=True,float_format='%0.2f',
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))})
print(res3)


# In[67]:


#regression summary table of all three regression outputs
res=summary_col([results4,results5,results6],stars=True,float_format='%0.2f',
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))})
print(res)

