def get_file_list(filePath='/home/svu/e0008216/Final_year_project/XenoCantoBirdCalls'):
    """
    This function returns a list of wav files in the specified folder
    """
    import os

    fileList = []
    for f in os.listdir(filePath):
        if f.endswith(".wav"):
            fileList.append(f)

    return fileList


def get_partitioned_data(wavData, rate, threshold=7000.0, length=2):
    """
    This function return a list of 2seconds data that are the noisiest
    It can be called a silence filterer
    Set a threshold for filter (default to be 1000)
    Also it normalizes the data automatically
    """
    import numpy as np
    # initialize dataList
    dataList = np.empty((0, rate*length), dtype=np.int8)
    # copy to retain wavData while iterating through it below
    copyWavData = wavData
    # suare up for easy threshold comparison
    squaredWavData = [x**2 for x in wavData]
    # ignoreThres is the number of samples passed// every 88200 samples
    ignoreThres = 0

    for i in range(len(squaredWavData)):
        # first occurence that crosses the threshold
        if squaredWavData[i] >= threshold**2 and i >= ignoreThres:

            #print("this occured at second: " + str(i*1.0/rate))
            ignoreThres = i+length*rate

            # HAVE TO RESHApE AAAAAA
            try:  # to check if end of audio file
                reshaped = copyWavData[i:i+length *
                                       rate:1].reshape(1, length*rate)
                # normalize the data here since its 16-bit => (-32768, 32767)
                reshaped = reshaped / 32768
                # append to the list
                dataList = np.append(dataList, reshaped, axis=0)
            except:
                continue

    return dataList


def get_features_and_labels(filePath='/home/svu/e0008216/Final_year_project/XenoCantoBirdCalls'):
    """
    This function combines all other functions to generate 2 lists
    1 is features_train and other is labels_train
    note that file name must be in the format:
    'abcxyz - label - abcxyz'
    so that we can extract the label
    """
    import numpy as np

    fileList = get_file_list(filePath)

    # harcoding the sampling rate
    features_train = np.empty((0, 44100*2), dtype=np.int8)
    labels_train = np.empty((0, 1), dtype=np.int8)

    count = 0
    labelList = []
    labelDict = {}
    featureList = []

    print("threshold is 7000")

    from scipy.io.wavfile import read

    for indx, f in enumerate(fileList):

        rate, wavData = read(filePath + "/" + f)
        print("generating for file: " + str(indx + 1) +
              " of " + str(len(fileList)))
        partitionedData = get_partitioned_data(wavData, rate, threshold=5000)
        features_train = np.append(features_train, partitionedData, axis=0)

        label = f.split(" - ")[1]  # get the middle one

        if label not in labelList:
            labelList.append(label)
            labelDict[labelList.index(label)] = (label, len(partitionedData))
        else:
            count = labelDict[labelList.index(label)][1]
            labelDict[labelList.index(label)] = (
                label, count + len(partitionedData))

        reshaped = np.array([labelList.index(label)]).reshape(1, 1)

        for _ in range(int(len(partitionedData))):
            labels_train = np.append(labels_train, reshaped, axis=0)
            featureList.append(f)

    return features_train, labels_train, labelList, labelDict, featureList


def save_features_and_labels(filePath='/home/svu/e0008216/Final_year_project/XenoCantoBirdCalls', savePath="/home/svu/e0008216/Final_year_project/featuresAndLabels"):
    import numpy as np
    import json

    features_train, labels_train, labelList, labelDict, featureList = get_features_and_labels(
        filePath)

    np.save(savePath + 'features.npy',
            np.array(features_train))
    np.save(savePath + 'labels.npy',
            np.array(labels_train))

    json.dump(labelDict, open(
        savePath + 'labelsDict.json', 'w'))
    json.dump(featureList, open(
        savePath + 'featureList.json', 'w'))
    json.dump(labelList, open(
        savePath + 'labelsList.json', 'w'))

    print("Features and Labels successfully extracted and saved")
    print("Features shape: ", features_train.shape)
    print("Labels shape: ", labels_train.shape)
    print("Number of identified birds:", len(np.unique(labels_train)))


if __name__ == "__main__":
    soundPath = "XenoCantoBirdCalls/"
    savePath = "FeaturesAndLabels/"
    save_features_and_labels(soundPath, savePath)
