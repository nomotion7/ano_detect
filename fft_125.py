# fft_125.py

import matplotlib.pyplot as plt
import data_parser_1 as dataParser
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

#filename = "1_FFT_Y_125_1_nok.csv"

#monitorIds, fftAxis, timestaps, frequencyArrays, amplitudeArrays = dataParser.parseFFT_IU(filename)

#print(len(monitorIds),len(fftAxis),len(timestaps), len(frequencyArrays),len(amplitudeArrays))



filename = "2_fft_140_4_test.csv"

timestamps, fftAxis, frequencyArrays, amplitudeArrays = dataParser.parseFFT_Treon(filename)


    



row = len(amplitudeArrays)-1
col = len(amplitudeArrays[0])
#print(row,frequencyArrays[50])

amp1 = np.zeros(row, dtype=float)
amp2 = np.zeros(row, dtype=float)
amp3 = np.zeros(row, dtype=float)
amp4 = np.zeros(row, dtype=float)
amp5 = np.zeros(row, dtype=float)
pseudo_time = np.zeros(row, dtype=float)
line = np.zeros(row, dtype=float)

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
    line[it1] = 1.1 #suggested threshold

bandwidth = 75


df2 = pd.DataFrame({'pseudo_time' : pseudo_time})

for it_1 in range(0,bandwidth):
    amp_it = np.zeros(row, dtype=float)
    for it_2 in range(0,row):
        amp_it[it_2] = amplitudeArrays[it_2][it_1]
    df2[it_1]= amp_it

#df2.info()


df2 = df2.drop(['pseudo_time'], axis=1)
names=df2.columns
x = df2[names]
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(x)

features = range(pca.n_components_)
""" _ = plt.figure(figsize=(25, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('Frequencies (sorted by PCA value)')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Principal Components Analysis on Frequency") """
#plt.show()

# Calculate PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])

df2['pc1']= principalDf['pc1']
df2['pc2']= principalDf['pc2']

#principalDf.info()

""" print(principalDf['pc1'])

from statsmodels.tsa.stattools import adfuller
# Run Augmented Dickey Fuller Test
result = adfuller(principalDf['pc1'])
# Print p-value
print(result[1]) """


# Assume that 13% of the entire data set are anomalies
outliers_fraction = 0.005

model =  IsolationForest(contamination=outliers_fraction)
model.fit(principalDf.values) 
principalDf['anomaly2'] = pd.Series(model.predict(principalDf.values))



# visualization
df2['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df2.index)
a = df2.loc[df2['anomaly2'] == -1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(principalDf['pc2'], color='blue', label='Normal')
_ = plt.plot(line, color='lightblue',linestyle='dotted',label='Threshhold')
_ = plt.plot(a['pc2'], linestyle='none', marker='X', color='red', markersize=12, label='Anomaly')
_ = plt.xlabel('Timestamps')
_ = plt.ylabel('FFT Amplitude')
_ = plt.title('Anomalies detected on frequency with heighest variance:'+filename)
_ = plt.legend(loc='best')
plt.show()


""" amp = np.zeros((row,bandwidth),dtype=float)

for it_freq in range(0,bandwidth):
    for it_amp in range(0,row):
        amp[it_amp][it_freq] = amplitudeArrays[it_amp][it_freq] """




""" for it1 in range(0,bandwidth):
    print(it1, frequencyArrays[0][it1]) """


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
