from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_125 = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\IU\Time-series\1_rms_125_1_Nok.csv')
df_125['date'] = pd.to_datetime(df_125['timestamp'])
del df_125['timestamp']

clustering1 = DBSCAN(eps = 0.05, min_samples = 3).fit(np.array(df_125['max_audio']).reshape(-1, 1))

labels = clustering1.labels_

outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['max_audio'])[pos])
    y.append(df_125['max_audio'].index[pos])

plt.plot(df_125['max_audio'].loc[df_125['max_audio'].index], 'k-')
plt.plot(y, x, 'r*', markersize=8)
plt.legend(['Actual', 'Anomaly Detected'])
plt.xlabel('Time Period')
plt.xticks([0, 20, 40, 60, 80, 99],
           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
            df_125.index[80], df_125.index[99]], rotation=45)
plt.ylabel('Maximal Audio')
plt.show()

max_x = df_125['max_vel_x']
max_y = df_125['max_vel_y']
max_z = df_125['max_vel_z']
#print(max_x)



clustering1 = DBSCAN(eps = 0.01, min_samples = 2).fit(np.array(df_125['max_vel_x']).reshape(-1, 1))

labels = clustering1.labels_

outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['max_vel_x'])[pos])
    y.append(df_125['max_vel_x'].index[pos])

plt.plot(df_125['max_vel_x'].loc[df_125['max_vel_x'].index], 'k-')
plt.plot(y, x, 'r*', markersize=8)
plt.legend(['Actual', 'Anomaly Detected'])
plt.xlabel('Time Period')
plt.xticks([0, 20, 40, 60, 80, 99],
           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
            df_125.index[80], df_125.index[99]], rotation=45)
plt.ylabel('Maximal X Difference')
plt.show()


def get_diff_cor(max_x):
    max_new_x = []
    for i in range(0, len(max_x)):
        if i == 0:
            max_new_x.append(0)
        elif i < len(max_x):
            max_new_x.append(max_x[i] - max_x[i + 1])
        elif i == len(max_x):
            max_new_x.append(max_x[i] - sum(max_x))
        elif i > len(max_x):
            break

    return max_new_x

max_x = get_diff_cor(max_x)
print(max_x)
max_y = get_diff_cor(max_y)
max_z = get_diff_cor(max_z)

df_125['cor_dif'] = max_x + max_y + max_y
print(df_125['cor_dif'])

clustering1 = DBSCAN(eps = 0.1, min_samples = 5).fit(np.array(df_125['cor_dif']).reshape(-1, 1))

labels = clustering1.labels_

outlier_pos = np.where(labels == -1)[0]

x = [];
y = [];
for pos in outlier_pos:
    x.append(np.array(df_125['cor_dif'])[pos])
    y.append(df_125['cor_dif'].index[pos])

plt.plot(df_125['cor_dif'].loc[df_125['cor_dif'].index], 'k-')
plt.plot(y, x, 'r*', markersize=8)
plt.legend(['Actual', 'Anomaly Detected'])
plt.xlabel('Time Period')
plt.xticks([0, 20, 40, 60, 80, 99],
           [df_125.index[0], df_125.index[20], df_125.index[40], df_125.index[60],
            df_125.index[80], df_125.index[99]], rotation=45)
plt.ylabel('Maximal X Difference')
plt.show()