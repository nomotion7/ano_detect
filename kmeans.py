import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster

df_ok = pd.read_excel(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\Treon\Time-series\2_rms_140_1_ok.xlsx')
print(df_ok)

plt.plot(df_ok['timestamp'], df_ok['Temperature'])
#plt.show()

df_nok = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\Treon\Time-series\2_rms_140_2_nok.csv')
print(df_nok)

plt.plot(df_nok['timestamp'], df_nok['Temperature'])
#plt.show()

df_125 = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\IU\Time-series\1_rms_125_1_ok.csv')
df_125['date'] = pd.to_datetime(df_125['timestamp'])
del df_125['timestamp']
#del df_125['machinename']
#del df_125['monitor_id']
print(df_125)

# Function that calculates the percentage of missing values
def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df), columns=['percent'])
    idx = nans['percent'] > 0
    return nans[idx]

print(calc_percent_NAs(df_125))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
# Extract the names of the numerical columns
#df_125_1 = df_125['max_vel_x', 'max_vel_y', 'max_vel_z', 'max_audio']
df2 = df_125.drop(['monitor_id', 'machinename', 'date', 'max_vel_x', 'max_vel_z', 'max_vel_y'], axis=1)
names=df2.columns
x = df2[names]
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)# Plot the principal components against their inertiafeatures = range(pca.n_components_)
features = range(pca.n_components_)
_ = plt.figure(figsize=(15, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Importance of the Principal Components based on inertia")
plt.show()

# Calculate PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

# Import necessary libraries
from sklearn.cluster import KMeans
# I will start k-means clustering with k=2 as I already know that there are 3 classes of "NORMAL" vs
# "NOT NORMAL" which are combination of BROKEN" and"RECOVERING"kmeans = KMeans(n_clusters=2, random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(principalDf.values)
labels = kmeans.predict(principalDf.values)
unique_elements, counts_elements = np.unique(labels, return_counts=True)
clusters = np.asarray((unique_elements, counts_elements))# Write a function that calculates distance between each point and the centroid of the closest cluster

def getDistanceByPoint(data, model):
    """ Function that calculates the distance between a point and centroid of a cluster, 
            returns the distances in pandas series"""
    distance = []
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.append(np.linalg.norm(Xa-Xb))
    return pd.Series(distance, index=data.index)
# Assume that 13% of the entire data set are anomalies
outliers_fraction = 0.24
# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(principalDf, kmeans)
# number of observations that equate to the 13% of the entire data set
number_of_outliers = int(outliers_fraction*len(distance))
# Take the minimum of the largest 13% of the distances as the threshold
threshold = distance.nlargest(number_of_outliers).min()
# anomaly1 contain the anomaly result of the above method Cluster (0:normal, 1:anomaly)
principalDf['anomaly1'] = (distance >= threshold).astype(int)

# visualization
df_125['anomaly1'] = pd.Series(principalDf['anomaly1'].values, index=df_125.index)
a = df_125.loc[df_125['anomaly1'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df_125['max_audio'], color='blue', label='Normal')
_ = plt.plot(a['max_audio'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Decible Reading')
_ = plt.title('Sensor Anomalies')
_ = plt.legend(loc='best')
plt.show();