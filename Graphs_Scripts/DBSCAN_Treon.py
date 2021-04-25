from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

df_125 = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\Treon\Time-series\2_rms_140_2_nok.csv')
df_125['date'] = pd.to_datetime(df_125['timestamp'])
del df_125['timestamp']

clustering1 = DBSCAN(eps = 0.05, min_samples = 3).fit(np.array(df_125['Temperature']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['Temperature'])[pos])
    y.append(df_125['Temperature'].index[pos])

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

Temperature = []
for i in range(0, len(df_125['Temperature'])):
    Temperature.append(i)

# Create traces
fig = go.Figure() #x="Decibel", y="Observations", color="Legend", title="Decible Anomaly Detection"
fig.add_trace(go.Scatter(x= Temperature, y=df_125['Temperature'], mode='lines', name='Sensor Celsius Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 29.3, line_width=3, line_dash="dash", line_color="green")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 2_rms_140_2_nok - Temperature",
        'yanchor': 'auto'},
    xaxis_title="Sensor Observations",
    yaxis_title="Temperature",
    legend_title="Legend")
fig.show()

max_x = []
for i in range(0, len(df_125['rms_x'])):
    max_x.append(i)
max_y = []
for i in range(0, len(df_125['rms_y'])):
    max_y.append(i)
max_z = []
for i in range(0, len(df_125['rms_z'])):
    max_z.append(i)
#print(max_x)

df_125['rms_x_1'] = [number / 100 for number in df_125['rms_x']]

clustering1 = DBSCAN(eps = 0.01, min_samples = 2).fit(np.array(df_125['rms_x_1']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['rms_x_1'])[pos])
    y.append(df_125['rms_x_1'].index[pos])

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_x, y=df_125['rms_x_1'], mode='lines', name='Sensor X Axis Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 2.5, line_width=3, line_dash="dash", line_color="green")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 2_rms_140_2_nok - X-Axis Vibration",
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

df_125['rms_y_1'] = [number / 100 for number in df_125['rms_y']]

clustering1 = DBSCAN(eps = 0.01, min_samples = 2).fit(np.array(df_125['rms_y_1']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['rms_y_1'])[pos])
    y.append(df_125['rms_y_1'].index[pos])

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_y, y=df_125['rms_y_1'], mode='lines', name='Sensor Y Axis Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 2.4, line_width=3, line_dash="dash", line_color="green")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 2_rms_140_2_nok - Y-Axis Vibration",
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

df_125['rms_z_1'] = [number / 100 for number in df_125['rms_z']]

clustering1 = DBSCAN(eps = 0.01, min_samples = 2).fit(np.array(df_125['rms_z_1']).reshape(-1, 1))
labels = clustering1.labels_
outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['rms_z_1'])[pos])
    y.append(df_125['rms_z_1'].index[pos])

# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x= max_z, y=df_125['rms_z_1'], mode='lines', name='Sensor Z Axis Measurements'))
fig.add_trace(go.Scatter(x=y, y=x, mode='markers', name='Anomalies'))
fig.update_traces(marker=dict(size=8))
fig.add_hline(y = 2.9, line_width=3, line_dash="dash", line_color="green")
fig.update_layout(
    title={
        'text': "Anomaly Detection - Sensor 2_rms_140_2_nok - Z-Axis Vibration",
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