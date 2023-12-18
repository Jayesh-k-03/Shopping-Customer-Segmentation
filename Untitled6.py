#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[6]:


df = pd.read_csv("C:/Users/jayesh/Downloads/Mall_Customers.csv")


# In[7]:


df.head()


# # univariate Analysis

# In[8]:


df.describe()


# In[10]:


sns.distplot(df['Annual Income (k$)']);


# In[19]:


df.columns


# In[26]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[49]:


sns.kdeplot(df['Annual Income (k$)'],shade=True);


# In[50]:


g = sns.FacetGrid(df, hue='Gender', height=6)
g.map(sns.kdeplot, 'Annual Income (k$)', shade=True).add_legend()

# Show the plot
plt.show()


# In[53]:


columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for col in columns:
    plt.figure()
    g = sns.FacetGrid(df, hue='Gender', height=6)
    g.map(sns.kdeplot, col, shade=True).add_legend()
    plt.title(f'KDE Plot for {col} by Gender')

plt.show()


# In[54]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[55]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[56]:


sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' )


# In[58]:


sns.pairplot(df,hue='Gender')


# In[59]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[60]:


df.corr()


# In[61]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# In[63]:


clustering1 = KMeans(n_clusters=3)


# In[64]:


clustering1.fit(df[['Annual Income (k$)']])


# In[66]:


clustering1.labels_


# In[67]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[71]:


df['Income Cluster'].value_counts()


# In[72]:


clustering1.inertia_


# In[74]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[75]:


intertia_scores


# In[76]:


plt.plot(range(1,11),intertia_scores)


# In[77]:


df.columns


# In[78]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # Bivariate Clustering

# In[79]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[80]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[81]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[82]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[83]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[84]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# # mulivariate clustering 

# In[85]:


from sklearn.preprocessing import StandardScaler


# In[86]:


scale = StandardScaler()


# In[87]:


df.head()


# In[88]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[89]:


dff.columns


# In[90]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[91]:


dff = scale.fit_transform(dff)


# In[92]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[93]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[94]:


df


# In[95]:


df.to_csv('Clustering.csv')


# In[ ]:




