import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import streamlit as st
#st.write("hh")

st.title("E-Commerce Data Segmentation")

df = pd.read_excel("cust_data.xlsx") 
st.subheader("Description of Data")
st.write(df.describe())

df[df.duplicated()]
st.subheader("Filling Null Values")
col1, col2 = st.columns(2)
with col1:
  st.write("Before")
  st.write(df.isna().sum())
  df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
with col2:
  st.write("After")
  st.write(df.isna().sum())

st.subheader("Gender Count")
st.write(df.Gender.value_counts())

sns.countplot(data=df,x='Gender')
plt.suptitle("Gender Count Graph")
#plt.show()


plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
sns.countplot(data=df, x='Orders')

plt.subplot(1,2,2)
sns.countplot(data=df, x='Orders', hue='Gender')
plt.suptitle("Overall Orders VS Gender wise Orders")
#plt.show()

##Boxplot for each Brand Orders
cols = list(df.columns[2:])
plt.suptitle("Boxplots")
def dist_list(lst):
  plt.figure(figsize=(30,30))
  for i, col in enumerate(lst,1):
    plt.subplot(6,6,i)
    sns.boxplot(data=df, x=df[col])
dist_list(cols)


##Heatmap
plt.figure(figsize=(20,15))
plt.suptitle("Heatmap")
sns.heatmap(df.iloc[:,3: ].corr())

##hist plot
df.iloc[:,2:].hist(figsize=(40,30))
plt.suptitle("History")

## create a new data set and total search column
new_df = df.copy()
st.subheader("Total Searches")
new_df['Total_Search'] = new_df.iloc[:,3:].sum(axis=1)
st.write(new_df.sort_values('Total_Search',ascending = False))

##Scaling
st.subheader("Scaling")
x = df.iloc[:,2:].values
st.write(x)
scale = MinMaxScaler()
features = scale.fit_transform(x)
st.write(features)


st.write("hh")
'''
##Silhouette Score for each k value
silhouette_avg = []
for i in range(2, 16):
  #initialize Kmeans
  kmeans = KMeans(n_clusters  =i)
  cluster_labels = kmeans.fit_predict(features)
  #silhouette Score
  silhouette_avg.append(silhouette_score(features, cluster_labels))

plt.figure(figsize=(10,7))
plt.plot(range(2,16),silhouette_avg, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette analysis for optimal K')
plt.show()
'''
st.write("hh")

##K-means Model
model = KMeans(n_clusters=3)
model = model.fit(features)
     
cm = model.predict(features)
centers = model.cluster_centers_

#creating cluster
new_df['Cluster'] = pd.DataFrame(cm)
new_df.to_csv('Cluster_data', index=False)

new_df['Cluster'].value_counts()
sns.countplot(data = new_df, x = 'Cluster')

##Analyzing Clusters 
c_df = pd.read_csv('Cluster_data')

##Analyzing Cluster 0 customer count and total searches graph
st.subheader("Cluster 0 data")
cl_0 = c_df.groupby(['Cluster', 'Gender'], as_index=False).sum().query('Cluster == 0')
cl_0
plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
sns.countplot(data = c_df.query('Cluster == 0'), x = 'Gender')
plt.title('Customers Count')

plt.subplot(1,2,2)
sns.barplot(data = cl_0, x = 'Gender', y = 'Total_Search')
plt.title('Total Searchs by Gender')
plt.suptitle('No. of Customers and their Total Searches in "Cluster 0"')

##Analyzing Cluster 1 customer count and total searches graph
st.subheader("Cluster 1 data")
cl_1 = c_df.groupby(['Cluster', 'Gender'], as_index=False).sum().query('Cluster == 1')
cl_1
plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
sns.countplot(data = c_df.query('Cluster == 1'), x = 'Gender')
plt.title('Customers Count')

plt.subplot(1,2,2)
sns.barplot(data = cl_1, x = 'Gender', y = 'Total_Search')
plt.title('Total Searchs by Gender')
plt.suptitle('No. of Customers and their Total Searches in "Cluster 1"')

## Analyzing Cluster 2 customer count and total searches graph
st.subheader("Cluster 2 data")
cl_2 = c_df.groupby(['Cluster', 'Gender'], as_index=False).sum().query('Cluster == 2')
cl_2
plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
sns.countplot(data = c_df.query('Cluster == 2'), x = 'Gender')
plt.title('Customers Count')

plt.subplot(1,2,2)
sns.barplot(data = cl_2, x = 'Gender', y = 'Total_Search')
plt.title('Total Searchs by Gender')
plt.suptitle('No. of Customers and their Total Searches in "Cluster 2"')

final_df = c_df.groupby(['Cluster'], as_index = False).sum()

#final data
plt.figure(figsize = (15,6))
sns.countplot(data = c_df, x='Cluster', hue = 'Gender')
plt.title('Total Customers on each Cluster')
plt.figure(figsize = (15,6))
plt.subplot(1,2,1)
sns.barplot(data = final_df, x = 'Cluster' , y = 'Total_Search')
plt.title('Total Searches by each group')

plt.subplot(1,2,2)
sns.barplot(data=final_df,x='Cluster',y='Orders')
plt.title('Past Orders by Each Group')
plt.suptitle('No. of times Customers Searched the products and their past Orders')
plt.show()





     

    


     



