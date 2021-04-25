from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import data_parser as dataParser
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df_125 = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\IU\Time-series\1_rms_125_2_test.csv')
df_125['date'] = pd.to_datetime(df_125['timestamp'])
del df_125['timestamp']

timestamps, fftAxis, frequencyArrays, amplitudeArrays = dataParser.parseFFT_Treon(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\Treon\FFT\2_fft_140_2_nok.csv')

print(amplitudeArrays)

clustering1 = DBSCAN(eps = 0.05, min_samples = 3).fit(np.array(amplitudeArrays[0]).reshape(-1, 1))

labels = clustering1.labels_

outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(amplitudeArrays[0])[pos])
    y.append(amplitudeArrays[0].index[pos])


max_audio = []
for i in range(0, len(amplitudeArrays[0])):
    max_audio.append(i)

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_audio, y=amplitudeArrays[0], mode='lines', name='lines'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='markers'))
fig.update_traces(marker=dict(size=8))
fig.show()