import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics


def parseFFT_IU(filename):
    # setup lists
    monitorIds = []
    fftAxis = []
    timestaps = []
    frequencyArrays = []
    amplitudeArrays = []

    # read fft
    with open(filename) as file:

        allChars = file.read()
        commaCounter = 0
        inArray = False
        currentValue = ""

        # iterate file char by char
        for i, c in enumerate(allChars):

            if c == ',' and not inArray:
                commaCounter += 1
                currentValue = ""
            elif c == '"':
                inArray = not inArray
            else:
                currentValue += c

            # add elements to lists
            if i < len(allChars) - 1 and allChars[i + 1] == ',' and not inArray or i == len(allChars) - 1:

                if commaCounter == 0:
                    monitorIds.append(currentValue)
                elif commaCounter == 1:
                    fftAxis.append(currentValue)
                elif commaCounter == 2:
                    timestaps.append(currentValue)
                elif commaCounter == 3:
                    frequencyArrays.append(currentValue.split(','))
                elif commaCounter == 4:
                    amplitudeArrays.append(currentValue.split(','))

            if commaCounter == 5:
                commaCounter = 0

            if len(monitorIds) == 10:
                break

    return monitorIds, fftAxis, timestaps, frequencyArrays, amplitudeArrays

df_125 = pd.read_csv(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\IU\FFT\1_FFT_Y_125_1_ok.csv')

df = parseFFT_IU(r'C:\Users\Max\Desktop\Hackathon_Data\Task 2 - Leadec - StarterKit\IU\FFT\1_FFT_Y_125_1_ok.csv')
print(df)
