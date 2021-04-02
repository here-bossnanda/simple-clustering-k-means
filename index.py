import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# READ DATA
data = pd.read_excel('data.xlsx', header=[0], index_col=None, na_values=['NA'])
data.head()
df = DataFrame(data,columns=['X','Y','Z'])


# FIND OPTIMAL K
mms = MinMaxScaler()
mms.fit(df)
data_transformed = mms.transform(df)

Sum_of_squared_distances = []
K = range(1,8)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# TRAIN DATA
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df["X"],df["Y"],df["Z"],c='black')
plt.scatter(centroids[:, 0],centroids[:, 1],centroids[:, 2],c='red')
plt.show()