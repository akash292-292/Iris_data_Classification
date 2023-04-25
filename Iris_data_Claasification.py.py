
#importing Libraries.
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plot





#import the data set Using Pandas
dataset=pd.read_csv('iris.csv')
#gives the first 10 rows of the dataset 
dataset.head(10)


# In[11]:


#gives last 5 rows of the dataset
dataset.tail()


# In[8]:


dataset.describe()


# In[12]:


#gives basic info about data type
dataset.info()


# In[16]:


#to display no samples on each class
dataset['class'].value_counts()


# In[17]:


print(dataset.groupby('class').size())





#Check for Null Values in the dataset
dataset.isnull().sum()





# Exploratary Data Analysis
#count on y-axis 
#values on x-axis
dataset['sepal.length'].hist()





dataset['sepal.width'].hist()





dataset['petal.length'].hist()





dataset['petal.width'].hist()





dataset.describe()


# In[32]:


#ScatterPlot
colors=['red','blue','black']
species=['Setosa','Virginica','Versicolor']


# In[36]:


for i in range(len(species)):
    x=dataset[dataset['class']==species[i]]
    plot.scatter(x['sepal.length'],x['sepal.width'],c=colors[i],label=species[i])
plot.xlabel('Sepal Length')    
plot.ylabel('Sepal Width')
plot.legend()


# In[37]:


for i in range(len(species)):
    x=dataset[dataset['class']==species[i]]
    plot.scatter(x['petal.length'],x['petal.width'],c=colors[i],label=species[i])
plot.xlabel('Petal Length')    
plot.ylabel('Petal Width')
plot.legend()


# In[46]:


dataset.corr()


# In[43]:


corr=dataset.corr()
fig,ax=plot.subplots(figsize=(5,5))
sns.heatmap(corr,annot=True,ax=ax)





#Label Encoding
#PreProcessing Technique 1->
#converting all datatypes to machine readable format
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()





dataset['class']=le.fit_transform(dataset['class'])
dataset.info()





dataset.describe()





from sklearn.model_selection import train_test_split
#train ->60
#testing->40
X=dataset.drop(columns=['class']);
Y=dataset['class']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.40)





from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()





model.fit(x_train,y_train)





print("Accuracy : ",model.score(x_test,y_test)*100)







