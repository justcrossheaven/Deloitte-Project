#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
# prepare dataset filling null values with 0

df = pd.read_excel("Deloitte Team 1 Debtor segmentation.xlsx")
df = df.fillna(0)
df_copy=df.copy()
df5 = pd.DataFrame(df, columns= ['Amount Due', 'Active Bucket','30<60','61<90','91<120','121+'])
model = KModes()
visualizer = KElbowVisualizer(model, k=(1,7))
visualizer.fit(df5)        # Fit the data to the visualizer
visualizer.show() 


# In[28]:


# binning by manually dividing the scale of amount into buckets and attaching labels to each bucket

df['Amount Due bin'] = pd.cut(df['Amount Due'], [-1,0,1000,3000,5000,10000,50000,1000000],
                         labels = ['0','1-1000', '1000-3000','3000-5000','5000-10000',
                                   '10000-50000','50000-1000000'])
df = df.drop('Amount Due', axis=1)
df['Active Bucket bin'] = pd.cut(df['Active Bucket'], [-1,0,1000,3000,5000,10000,20000],
                         labels = ['0','1-1000', '1000-3000','3000-5000',
                                   '5000-10000','10000-20000'])
df = df.drop('Active Bucket', axis=1)
df['30<60 bin'] = pd.cut(df['30<60'], [-1,0,1000,3000,5000,10000,20000],
                         labels = ['0','1-1000', '1000-3000','3000-5000',
                                   '5000-10000','15000-20000'])
df = df.drop('30<60', axis=1)
df['61<90 bin'] = pd.cut(df['61<90'], [-1,0,1000,3000,5000,10000,40000],
                         labels = ['0','1-1000', '1000-3000','3000-5000',
                                   '5000-10000','15000-40000'])
df = df.drop('61<90', axis=1)
df['91<120 bin'] = pd.cut(df['91<120'], [-1,0,1000,3000,5000,10000,100000],
                         labels = ['0','1-1000', '1000-3000','3000-5000',
                                   '5000-10000','10000-100000'])
df = df.drop('91<120', axis=1)
df['121+ bin'] = pd.cut(df['121+'], [-1,0,1000,3000,5000,10000,200000],
                         labels = ['0','1-1000', '1000-3000','3000-5000',
                                   '5000-10000','10000-200000'])
df = df.drop('121+', axis=1)


# In[29]:


# create a sub-dataset containing only columns related to amount due
# which are more relevant to the risk level clustering
df1 = pd.DataFrame(df, columns= ['Amount Due bin', 'Active Bucket bin','30<60 bin','61<90 bin','91<120 bin','121+ bin'])


# fit the KModes model with k=4
kmodes = KModes(n_clusters = 3)
clusters = kmodes.fit_predict(df1)


# attach the cluster prediction to original dataset
clusters = pd.DataFrame(clusters)
clusters.columns = ['cluster_pred']
combined_df = pd.concat([df, clusters], axis=1).reset_index()
combined_df = combined_df.drop(['index'],axis=1)


# Print the cluster centroids
print(kmodes.cluster_centroids_)


# print cluster labels
print(clusters)


# print out the number of datapoints in each cluster
print(combined_df['cluster_pred'].value_counts())


# In[30]:


combined_df.to_excel(r'C:\Users\chenh\Downloads\newoutputresult.xlsx')


# In[31]:


fig = plt.figure(figsize = (40,20))

ax1 = fig.add_subplot(111)
ax1.set_title('risk analysis')

plt.xlabel('Amount Due')
plt.ylabel('risk level cluster')

x = df_copy['Amount Due']
y = combined_df['cluster_pred']

plt.xticks([0,5000,10000,20000,30000,50000,350000])
plt.yticks([0,1,2,3])
plt.text(336677,3,'Emirates Airlines')
plt.text(58543,3,'Air NZ')
plt.text(63938,1,'Singapore Airlines')

plt.scatter(x,y)
plt.savefig("scatter.png")
plt.show()


# In[ ]:




