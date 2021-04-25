# fft_kmeanscluster.py

import matplotlib.pyplot as plt
import data_parser as dataParser
#import plotter as plot
import numpy as np
from numpy import ndarray
import pandas as pd
import sklearn.cluster


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from scipy.stats import iqr

from sklearn.ensemble import IsolationForest

#plot.plotFFT_Treon("2_fft_140_4_test.csv")


#Run augemented dickey fuller test

#frequencyArrays[0]

filename = "2_fft_140_2_ok.csv"

timestamps, fftAxis, frequencyArrays, amplitudeArrays = dataParser.parseFFT_Treon(filename)

#save parsed values 
#parsed = pd.DataFrame({'time': timestamps,'freq': frequencyArrays, 'amp': amplitudeArrays})

#parsed.to_csv('parsed_2_fft_140_2_ok.csv', header = False)

row = len(amplitudeArrays)-1
col_numb_freq = len(frequencyArrays)
#print(row,frequencyArrays[50])

amp1 = np.zeros(row, dtype=float)
amp2 = np.zeros(row, dtype=float)
amp3 = np.zeros(row, dtype=float)
amp4 = np.zeros(row, dtype=float)
amp5 = np.zeros(row, dtype=float)
pseudo_time = np.zeros(row, dtype=float)

freq1 = 0 #hz
freq2 = 1 #hz
freq3 = 49 #hz
freq4 = 50 #hz
freq5 = 51 #hz 



for it1 in range(0,row):
    amp1[it1] = amplitudeArrays[it1][freq1]
    amp2[it1] = amplitudeArrays[it1][freq2]
    amp3[it1] = amplitudeArrays[it1][freq3]
    amp4[it1] = amplitudeArrays[it1][freq4]
    amp5[it1] = amplitudeArrays[it1][freq5]
    pseudo_time[it1] = it1


#plotting amplitude at 5 frequencies 

""" plt.rcParams["figure.figsize"] = (20,15)

plt.subplot(5, 1, 1)
plt.plot(pseudo_time, amp1, label= "0 hz")

plt.subplot(5, 1, 2)
plt.plot(pseudo_time, amp2, label= "1 hz")

plt.subplot(5, 1, 3)
plt.plot(pseudo_time, amp3, label= "49 hz")

plt.subplot(5, 1, 4)
plt.plot(pseudo_time, amp4, label= "50 hz")

plt.subplot(5, 1, 5)
plt.plot(pseudo_time, amp5, label= "51 hz")

#plt.plot(pseudo_time, amp1, label="Freq 50")
plt.show() """


""" print( amplitudeArrays[0][:] )

null_count = 0 
percentile = np.zeros(len(amplitudeArrays), dtype=float)

for it_2 in range(0,len(amplitudeArrays[:][0])):

    for it_1 in amplitudeArrays[0]:
        if it_1== 0:
            null_count = null_count +1

    percentile[it_2] = null_count/len(amplitudeArrays[0])
    null_count = 0

print ( len(amplitudeArrays[0]) , null_count)

print (percentile) """












df2 = pd.DataFrame({'amp1' : amp1, 'amp2' : amp2})



names= df2.columns
x = df2[names]
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)# Plot the principal components against their inertiafeatures = range(pca.n_components_)
features = range(pca.n_components_)
""" _ = plt.figure(figsize=(15, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Importance of the Principal Components based on inertia") """
#plt.show()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df2)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

df2['pc1']= principalDf['pc1']
df2['pc2']= principalDf['pc2']


outliers_fraction = 0.001

model =  IsolationForest(contamination=outliers_fraction)
model.fit(principalDf.values) 
principalDf['anomaly2'] = pd.Series(model.predict(principalDf.values))

df2['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df2.index)
a = df2.loc[df2['anomaly2'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df2['amp1'], color='blue', label='Normal')
_ = plt.plot(a['amp1'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('timesteps')
_ = plt.ylabel('freq amplitude')
_ = plt.title('freq anomalies detected with isolation forrest')
_ = plt.legend(loc='best')
plt.show()










""" lower_bound = 0.005; 
upper_bound = 0.995; 

# Calculate IQR for the 1st principal component (pc1)

q1_pc1, q3_pc1 = principalDf['pc1'].quantile([lower_bound, upper_bound])
iqr_pc1 = q3_pc1 - q1_pc1


# Calculate upper and lower bounds for outlier for pc1
lower_pc1 = q1_pc1 - (1.5*iqr_pc1)
upper_pc1 = q3_pc1 + (1.5*iqr_pc1)

# Filter out the outliers from the pc1
df2['anomaly_pc1'] = ((df2['pc1']>upper_pc1) | (df2['pc1']<lower_pc1))#.astype('float')

# Calculate IQR for the 2nd principal component (pc2)
q1_pc2, q3_pc2 = df2['pc2'].quantile([lower_bound, upper_bound])

iqr_pc2 = q3_pc2 - q1_pc2

# Calculate upper and lower bounds for outlier for pc2
lower_pc2 = q1_pc2 - (1.5*iqr_pc2)
upper_pc2 = q3_pc2 + (1.5*iqr_pc2)

# Filter out the outliers from the pc2
df2['anomaly_pc2'] = ((df2['pc2']>upper_pc2) | (df2['pc2']<lower_pc2))#.astype('int')

# Let's plot the outliers from pc1 on top of the sensor_11 and see where they occured in the time series
a = df2[df2['anomaly_pc1'] == 1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(df2['amp1'], color='blue', label='Normal')
_ = plt.plot(a['amp1'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Amplitude')
_ = plt.title('amp1 Anomalies')
_ = plt.legend(loc='best')
plt.show() """