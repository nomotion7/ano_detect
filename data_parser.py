import csv

def parseRMS_IU(filename):

    # setup lists
    timestaps = []
    monitorIds = []
    machinenames = []
    maxVelXs = []
    maxVelYs = []
    maxVelZs = []
    maxAudios = []

    # read rms
    with open(filename) as file:
        print("Parsing RMS")

        csvReader = csv.reader(file, delimiter=',')
        lineCounter = 0
        for row in csvReader:
            
            if lineCounter == 0:
                lineCounter += 1
                continue

            # fill lists
            timestaps.append(row[0])
            monitorIds.append(row[1])
            machinenames.append(row[2])
            maxVelXs.append(row[3])
            maxVelYs.append(row[4])
            maxVelZs.append(row[5])
            maxAudios.append(row[6])

            lineCounter += 1

            if lineCounter == 200:
                break
    
    return timestaps, monitorIds, machinenames, maxVelXs, maxVelYs, maxVelZs, maxAudios
            
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
            if i < len(allChars)-1 and allChars[i+1] == ',' and not inArray or i == len(allChars)-1:
                
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

def parseRMS_Treon(filename):

    # setup lists
    timestaps = []
    sensorNodeIds = []
    temperatures = []
    rmsXs = []
    rmsYs = []
    rmsZs = []

    # read rms
    with open(filename) as file:
        print("Parsing RMS")

        csvReader = csv.reader(file, delimiter=',')
        lineCounter = 0
        for row in csvReader:
            
            if lineCounter == 0:
                lineCounter += 1
                continue

            # fill lists
            timestaps.append(row[0])
            sensorNodeIds.append(row[1])
            temperatures.append(row[2])
            rmsXs.append(row[3])
            rmsYs.append(row[4])
            rmsZs.append(row[5])

            lineCounter += 1

            # if lineCounter == 10:
            #     break
    
    return timestaps, sensorNodeIds, temperatures, rmsXs, rmsYs, rmsZs
    
def parseFFT_Treon(filename):

    # setup lists
    timestamps = []
    fftAxis = []
    frequencyArrays = []
    amplitudeArrays = []

    # read fft
    with open(filename) as file:

        allChars = file.read()
        commaCounter = 0
        inArray = False
        inArea = False
        currentValue = ""

        
        # iterate file char by char
        for i, c in enumerate(allChars):
            
            if c.isdigit() and not inArea:
                inArea = True 

            if inArea:
                # separate elements
                if c == ',' and not inArray:
                    commaCounter += 1
                    currentValue = ""
                elif c == '"':
                    inArray = not inArray
                else:
                    currentValue += c
                
                # add elements to lists
                if i < len(allChars)-1 and allChars[i+1] == ',' and not inArray or i == len(allChars)-1:
                    
                    if commaCounter == 0:
                        timestamps.append(currentValue)
                    elif commaCounter == 1:
                        fftAxis.append(currentValue)
                    elif commaCounter == 2:
                        amplitudeArrays.append(currentValue.split(','))
                        frequencyArrays.append(list(range(0, len(amplitudeArrays[-1]))))
                

                if commaCounter == 3:
                    commaCounter = 0
                
                # if len(timestamps) == 10:
                #     break


    return timestamps[1:None], fftAxis, frequencyArrays, amplitudeArrays


# parseFFT_Treon("F:/Users/Lukas/Documents/Hackathon2021/Task 2 - Leadec - StarterKit/Treon/FFT/2_fft_140_1_ok.csv")