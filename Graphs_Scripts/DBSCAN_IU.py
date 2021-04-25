from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df_125 = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\IU\Time-series\1_rms_125_2_test.csv')
df_125['date'] = pd.to_datetime(df_125['timestamp'])
del df_125['timestamp']

clustering1 = DBSCAN(eps = 0.05, min_samples = 3).fit(np.array(df_125['max_audio']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(labels)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['max_audio'])[pos])
    y.append(df_125['max_audio'].index[pos])

#plt.plot(df_125['Temperature'].loc[df_125['Temperature'].index], 'k-')
#plt.plot(y, x, 'r*', markersize=8)
#plt.legend(['Actual', 'Anomaly Detected'])
#plt.xlabel('Time Period')
#plt.xticks([0, 20, 40, 60, 80, 99],
#           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
#            df_125.index[80], df_125.index[99]], rotation=45)
#plt.ylabel('Maximal Audio')
#plt.show()

#fig = px.line(df_125['Temperature'].loc[df_125['Temperature'].index])
#fig.add_trace(go.Scatter(x = y, y = x, mode='markers', name='markers'))
#fig.update_traces(marker=dict(size=8))
#fig.show()

max_audio = []
for i in range(0, len(df_125['max_audio'])):
    max_audio.append(i)

# Create traces
fig = go.Figure() #x="Decibel", y="Observations", color="Legend", title="Decible Anomaly Detection"
fig.add_trace(go.Scatter(x= max_audio, y=df_125['max_audio'], mode='lines', name='Sensor Decibel Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 109, line_width=3, line_dash="dash", line_color="green", annotation_text ='Border for ~95% of Data', annotation_position="top left")
fig.add_hline(y = 109*1.07, line_width=3, line_dash="dot", line_color="red", annotation_text ='Established Threshold of Functionality Critical Values', annotation_position="top left")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 1_rms_125_2_test - Audio",
        'yanchor': 'auto'},
    xaxis_title="Sensor Observations",
    yaxis_title="Audio",
    legend_title="Legend")
fig.show()

max_x = []
for i in range(0, len(df_125['max_vel_x'])):
    max_x.append(i)
max_y = []
for i in range(0, len(df_125['max_vel_y'])):
    max_y.append(i)
max_z = []
for i in range(0, len(df_125['max_vel_z'])):
    max_z.append(i)
#print(max_x)


clustering1 = DBSCAN(eps = 0.01, min_samples = 4).fit(np.array(df_125['max_vel_x']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['max_vel_x'])[pos])
    y.append(df_125['max_vel_x'].index[pos])

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_x, y=df_125['max_vel_x'], mode='lines', name='Sensor X Axis Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 2.3, line_width=3, line_dash="dash", line_color="green", annotation_text ='Border for ~95% of Data', annotation_position="top left")
fig.add_hline(y = 2.7, line_width=3, line_dash="dot", line_color="red", annotation_text ='Established Threshold of Functionality Critical Values', annotation_position="top left")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 1_rms_125_2_test - X-Axis Vibration",
        'yanchor': 'auto'},
    xaxis_title="Sensor Observations",
    yaxis_title="X Axis Vibration in MM/S",
    legend_title="Legend")
fig.show()


#plt.plot(df_125['rms_x'].loc[df_125['rms_x'].index], 'k-')
#plt.plot(y, x, 'r*', markersize=8)
#plt.legend(['Actual', 'Anomaly Detected'])
#plt.xlabel('Time Period')
#plt.xticks([0, 20, 40, 60, 80, 99],
#           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
#            df_125.index[80], df_125.index[99]], rotation=45)
#plt.ylabel('Maximal X Difference')
#plt.show()


#def get_diff_cor(max_x):
#    max_new_x = []
#    for i in range(0, len(max_x)):
#        if i == 0:
#            max_new_x.append(0)
#        elif i < len(max_x):
#            max_new_x.append(max_x[i] - max_x[i + 1])
#        elif i == len(max_x):
#            max_new_x.append(max_x[i] - sum(max_x))
#        elif i > len(max_x):
#            break
#
#    return max_new_x

#max_x = get_diff_cor(max_x)
#print(max_x)
#max_y = get_diff_cor(max_y)
#max_z = get_diff_cor(max_z)

#df_125['cor_dif'] = max_x + max_y + max_y
#print(df_125['cor_dif'])

clustering1 = DBSCAN(eps = 0.01, min_samples = 4).fit(np.array(df_125['max_vel_y']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['max_vel_y'])[pos])
    y.append(df_125['max_vel_y'].index[pos])

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_y, y=df_125['max_vel_y'], mode='lines', name='Sensor Y Axis Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 5.2, line_width=3, line_dash="dash", line_color="green", annotation_text ='Border for ~95% of Data', annotation_position="top left")
fig.add_hline(y = 5.2*1.4, line_width=3, line_dash="dot", line_color="red", annotation_text ='Established Threshold of Functionality Critical Values', annotation_position="top left")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 1_rms_125_2_test - Y-Axis Vibration",
        'yanchor': 'auto'
        },
    xaxis_title="Sensor Observations",
    yaxis_title="Y Axis Vibration in MM/S",
    legend_title="Legend")
fig.show()

#plt.plot(df_125['rms_y'].loc[df_125['rms_y'].index], 'k-')
#plt.plot(y, x, 'r*', markersize=8)
#plt.legend(['Actual', 'Anomaly Detected'])
#plt.xlabel('Time Period')
#plt.xticks([0, 20, 40, 60, 80, 99],
#           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
#            df_125.index[80], df_125.index[99]], rotation=45)
#plt.ylabel('Maximal X Difference')
#plt.show()

clustering_z = DBSCAN(eps = 0.01, min_samples = 4).fit(np.array(df_125['max_vel_z']).reshape(-1, 1))
labels_z = clustering_z.labels_
outlier_pos = np.where(labels_z == -1)[0]

n_clusters_ = len(set(labels_z)) - (1 if -1 in labels else 0)
n_noise_ = list(labels_z).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['max_vel_z'])[pos])
    y.append(df_125['max_vel_z'].index[pos])

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_z, y=df_125['max_vel_z'], mode='lines', name='Sensor Z Axis Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 9, line_width=3, line_dash="dash", line_color="green", annotation_text ='Border for ~95% of Data', annotation_position="top left")
fig.add_hline(y = 9*1.2, line_width=3, line_dash="dot", line_color="red", annotation_text ='Established Threshold of Functionality Critical Values', annotation_position="top left")
fig.update_layout(
    title={
        'text': "Anomaly Detection - 1_rms_125_2_test - Z-Axis Vibration",
        'yanchor': 'auto'
        },
    xaxis_title="Sensor Observations",
    yaxis_title="Z Axis Vibration in MM/S",
    legend_title="Legend")
fig.show()

#plt.plot(df_125['rms_z'].loc[df_125['rms_z'].index], 'k-')
#plt.plot(y, x, 'r*', markersize=8)
#plt.legend(['Actual', 'Anomaly Detected'])
#plt.xlabel('Time Period')
#plt.xticks([0, 20, 40, 60, 80, 99],
#           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
#            df_125.index[80], df_125.index[99]], rotation=45)
#plt.ylabel('Maximal X Difference')
#plt.show()