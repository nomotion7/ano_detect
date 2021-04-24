import matplotlib.pyplot as plt
import data_parser as dataParser

def plotRMS_IU(filename):

    timestaps, monitorIds, machinenames, maxVelXs, maxVelYs, maxVelZs, maxAudios = dataParser.parseRMS_IU(filename)

    fig, ax = plt.subplots()
    ax.plot(timestaps, maxVelXs, label="VelX")
    fig.suptitle("Velocities X")
    plt.show()

def plotRMS_Treon(filename):

    timestaps, sensorNodeIds, temperatures, rmsXs, rmsYs, rmsZs = dataParser.parseRMS_Treon(filename)


    plt.rcParams["figure.figsize"] = (20,10)

    plt.subplot(4, 1, 1)
    plt.plot(timestaps, temperatures, label="Temperatures")

    
    plt.subplot(4, 1, 2)
    plt.plot(timestaps, rmsXs, label="RMS X")

    
    plt.subplot(4, 1, 3)
    plt.plot(timestaps, rmsYs, label="RMS Y")

    
    plt.subplot(4, 1, 4)
    plt.plot(timestaps, rmsZs, label="RMS Z")

    plt.suptitle("RMS Treon - " + filename.split("/")[-1])

    plt.show()

def plotFFT_Treon(filename):

    timestamps, fftAxis, frequencyArrays, amplitudeArrays = dataParser.parseFFT_Treon(filename)

    plt.rcParams["figure.figsize"] = (20,10)

    plt.subplot(4, 1, 1)
    plt.plot(frequencyArrays[0], amplitudeArrays[0], label="Amplitude 0")

    plt.subplot(4, 1, 2)
    plt.plot(frequencyArrays[1], amplitudeArrays[1], label="Amplitude 1")

    plt.subplot(4, 1, 3)
    plt.plot(frequencyArrays[2], amplitudeArrays[2], label="Amplitude 2")

    plt.subplot(4, 1, 4)
    plt.plot(frequencyArrays[3], amplitudeArrays[3], label="Amplitude 3")

    plt.suptitle("FFT Treon - " + filename.split("/")[-1])

    plt.show()


# plotRMS_Treon("F:/Users/Lukas/Documents/Hackathon2021/Task 2 - Leadec - StarterKit/Treon/Time-series/")
plotFFT_Treon("F:/Users/Lukas/Documents/Hackathon2021/Task 2 - Leadec - StarterKit/Treon/FFT/2_fft_140_2_nok.csv")





