import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
import random

# Библиотеки поиска максимумов
from scipy import signal
from scipy.signal import argrelmax


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Conv1D, Conv2D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# import all libraries
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



'''Функция открытия файла с датасетом'''
def fileOpen(path):

    with open(path, "r") as read_file:
        data = json.load(read_file)
    data = np.asarray(data)
    print("Размер набора данных составляет " + str(data.shape))
    return data


'''
Функция анализа пустоты на этапе калибровки 
Описание: Сюда подаём часть датасета, размер которой равен количеству калибровочных пакетов, т.е. пакетов без движения
'''
def defineTreshold(mass):
    print(mass.shape)
    aveCalibration = np.average(mass)
    maxCalibration = np.max(mass)
    minCalibration = np.min(mass)
    print("Ave = "+str(aveCalibration))
    print("Min = "+str(minCalibration))
    print("Max = "+str(maxCalibration))
    #return aveCalibration
    return aveCalibration*2

'''
Функция вычитания порогового значения tres из элементов массива mass и возвращение его абсолютных значений
'''
def treshold(mass, tres):
    for i in range(len(mass)):
        if (abs(mass[i])<tres):
            mass[i]=0
        else:
            if mass[i]>0:
                mass[i] = mass[i] - tres
            if mass[i] < 0:
                mass[i] = mass[i] + tres
    return mass
    #return abs(mass)

def KalmanFilterSimple1D(data, q, r, state, covariance, f=1, h=1):
        # Инициализация данных
        Q = q
        R = r
        F = f
        H = h
        State = state
        Covariance = covariance

        # time update - prediction
        X0 = F * State
        P0 = F * Covariance * F + Q
        # measurement update - correction
        K = H * P0 / (H * P0 * H + R)
        State = X0 + K * (data - H * X0)
        Covariance = (1 - K * H) * P0
        return State

def kalmanFilter(mass):
    KalmanFiltered = []
    curState = mass[0]
    for i in range(len(mass)):
        curState = KalmanFilterSimple1D(data=mass[i], q=0.3, r=5, state=curState, covariance=0.1,
                                        f=1,
                                        h=1)  # r = 3 - что-то типа шага. Чем больше, тем грубее                                                                                                             # covariance на сколько гасить
        KalmanFiltered.append(curState)
    KalmanFiltered = np.asarray(KalmanFiltered)
    return KalmanFiltered

def viewResults5x2Plot(mass1, mass2, mass3, mass4, mass5, mass6, mass7, mass8, mass9, mass10):
    plt.figure(figsize=(14, 9))

    plt.subplot(521)
    plt.plot(mass1, 'r')


    plt.subplot(522)
    plt.plot(mass2, 'r')

    plt.subplot(523)
    plt.plot(mass3)

    plt.subplot(524)
    plt.plot(mass4)

    plt.subplot(525)
    plt.plot(mass5)

    plt.subplot(526)
    plt.plot(mass6)

    plt.subplot(527)
    plt.plot(mass7)

    plt.subplot(528)
    plt.plot(mass8)

    plt.subplot(529)
    #Для пиков
    #plt.scatter(mass9, range(len(mass9)))
    #Для графика
    plt.plot(mass9)
    #plt.subplot(5210)
    #plt.plot(mass10)

    plt.show()
def viewResults5x2PlotNew(mass1, mass2, mass3, mass4, mass5, mass6, mass7, mass8, mass9, mass10, mass11, mass12):
    plt.figure(figsize=(14, 9))
    plt.subplot2grid((6, 2), (0, 0))
    plt.plot(mass1)
    plt.subplot2grid((6, 2), (0, 1))
    plt.plot(mass2)
    plt.subplot2grid((6, 2), (1, 0))
    plt.plot(mass3)
    plt.subplot2grid((6, 2), (1, 1))
    plt.plot(mass4)
    plt.subplot2grid((6, 2), (2, 0))
    plt.plot(mass5)
    plt.subplot2grid((6, 2), (2, 1))
    plt.plot(mass6)
    plt.subplot2grid((6, 2), (3, 0))
    plt.plot(mass7)
    plt.subplot2grid((6, 2), (3, 1))
    plt.plot(mass8)
    plt.subplot2grid((6, 2), (4, 0))
    plt.plot(mass9)
    plt.subplot2grid((6, 2), (4, 1))
    plt.plot(mass10)
    plt.subplot2grid((6, 2), (5, 0))
    plt.scatter(mass11, range(len(mass11)))
    plt.subplot2grid((6, 2), (5, 1))
    plt.scatter(mass12, range(len(mass12)))

    plt.show()

def voidDetectorKalmanPCA(dataToAnalyze):
        # Применяем метод главных компонент

        print("Поступили данные размером "+str(dataToAnalyze.shape))
        kalmanData = np.zeros((56, len(dataToAnalyze[0])))
        for i in range(56):
            kalmanData[i] = kalmanFilter(dataToAnalyze[i])

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(kalmanData.T).T

        # Первый главный компонент
        firstPCA = kalmanFilter(X_pca[1])
        # Второй главный компонент
        secondPCA = kalmanFilter(X_pca[2])


        # Определение калибровочного коэффициента
        numOfCalibrationPackets = 3000
        calibrationCoef = defineTreshold(abs(X_pca[1][0:numOfCalibrationPackets]))

        # Вычитание калибровочного коэффициента из сигнала
        # Первый главный компонент
        firstTresPCA = treshold(firstPCA, calibrationCoef)
        # Второй главный компонент
        secondTresPCA = treshold(secondPCA, calibrationCoef)


        '''
        # Визуализация результатов PCA для статьи (возможно)
        # visualizePCA2D(X_pca.T)

        

        # Определение калибровочного коэффициента
        numOfCalibrationPackets = 3000
        calibrationCoef = defineTreshold(abs(X_pca[1][0:numOfCalibrationPackets]))

        # Домножаем на 1.5 - экспериментально
        # Причина: Горизонтальные движения в верхней части наблюдаемого помещения даюит малые высокочастотные помехи
        calibrationCoef = calibrationCoef

        # Сортировка величин по убыванию
        # sortedMass1 = sorted(treshold(firstPCA, calibrationCoef), reverse=True)
        # sortedMass2 = sorted(treshold(secondPCA, calibrationCoef), reverse=True)

        # Вычитание калибровочного коэффициента из сигнала
        # Первый главный компонент
        firstTresPCA = treshold(firstPCA, calibrationCoef * 2)
        # Второй главный компонент
        secondTresPCA = treshold(secondPCA, calibrationCoef)

        # Обработка медианным фильтром
        firstMedTresPCA = signal.medfilt(firstTresPCA)
        secondMedTresPCA = signal.medfilt(secondTresPCA)

        medPCA1mass2D = np.stack((firstMedTresPCA, range(len(firstMedTresPCA))))
        medPCA2mass2D = np.stack((secondMedTresPCA, range(len(secondMedTresPCA))))

        # Фильтруем данные Калманом
        firstPCAKalmanFiltered = kalmanFilter(firstMedTresPCA)

        # Фильтруем данные Калманом
        SecondPCAKalmanFiltered = kalmanFilter(secondMedTresPCA)
        '''
        # Поиск максимумов https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html#scipy.signal.argrelmax
        # Производим поиск локальных максимумов сигнала


        stepSize=200
        fullity=[]
        for i in range(int(len(firstTresPCA)/stepSize)):
            fullity.append(np.sum(np.abs(firstTresPCA[i*stepSize:i*stepSize+stepSize])))





        firstMaxesMedTresPCA = np.asarray(argrelmax(firstTresPCA)).T
        secondMaxesMedTresPCA = np.asarray(argrelmax(secondTresPCA)).T

        # Выводим информацию на графики 5х2
        viewResults5x2PlotNew(dataToAnalyze[0], dataToAnalyze[0],
                              X_pca[1], X_pca[2],
                              firstPCA, secondPCA,
                              firstTresPCA, secondTresPCA,
                              fullity, secondMaxesMedTresPCA,
                              firstMaxesMedTresPCA, secondMaxesMedTresPCA)




def voidDetector(dataToAnalyze):
        # Применяем метод главных компонент
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(dataToAnalyze.T).T

        # Визуализация результатов PCA для статьи (возможно)
        #visualizePCA2D(X_pca.T)

        # Первый главный компонент
        firstPCA = kalmanFilter(X_pca[1])
        # Второй главный компонент
        secondPCA = kalmanFilter(X_pca[2])

        # Определение калибровочного коэффициента
        numOfCalibrationPackets = 3000
        calibrationCoef = defineTreshold(abs(X_pca[1][0:numOfCalibrationPackets]))

        # Домножаем на 1.5 - экспериментально
        # Причина: Горизонтальные движения в верхней части наблюдаемого помещения даюит малые высокочастотные помехи
        calibrationCoef = calibrationCoef

        # Сортировка величин по убыванию
        # sortedMass1 = sorted(treshold(firstPCA, calibrationCoef), reverse=True)
        # sortedMass2 = sorted(treshold(secondPCA, calibrationCoef), reverse=True)

        # Вычитание калибровочного коэффициента из сигнала
        # Первый главный компонент
        firstTresPCA = treshold(firstPCA, calibrationCoef)
        # Второй главный компонент
        secondTresPCA = treshold(secondPCA, calibrationCoef)

        # Обработка медианным фильтром
        firstMedTresPCA = signal.medfilt(firstTresPCA)
        secondMedTresPCA = signal.medfilt(secondTresPCA)

        medPCA1mass2D = np.stack((firstMedTresPCA, range(len(firstMedTresPCA))))
        medPCA2mass2D = np.stack((secondMedTresPCA, range(len(secondMedTresPCA))))

        # Фильтруем данные Калманом
        firstPCAKalmanFiltered = kalmanFilter(firstMedTresPCA)

        # Фильтруем данные Калманом
        SecondPCAKalmanFiltered = kalmanFilter(secondMedTresPCA)

        # Поиск максимумов https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html#scipy.signal.argrelmax
        # Производим поиск локальных максимумов сигнала
        firstMaxesMedTresPCA = np.asarray(argrelmax(firstPCAKalmanFiltered)).T
        secondMaxesMedTresPCA = np.asarray(argrelmax(SecondPCAKalmanFiltered)).T

        # Выводим информацию на графики 5х2
        viewResults5x2PlotNew(dataToAnalyze[0], dataToAnalyze[0], firstPCA, secondPCA, firstTresPCA,
                              secondTresPCA, firstMedTresPCA, secondMedTresPCA,
                              firstPCAKalmanFiltered, SecondPCAKalmanFiltered, firstMaxesMedTresPCA,
                              secondMaxesMedTresPCA)

        # Поиск пиков https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        '''
        peaks, _ = find_peaks(medPCA1mass, height=0)
        plt.plot(medPCA1mass)
        plt.plot(peaks, medPCA1mass[peaks], "x")
        plt.plot(np.zeros_like(medPCA1mass), "--", color="gray")
        plt.show()
        '''

def loadDataSet(path):

        data = fileOpen(path)

        # Изменение представления данных на (коли-во пакетов, кол-во антенн, кол-во поднесущих)
        dataToAnalyze = []
        countPackets = len(data)
        for i in range(countPackets):
            line = []
            for j in range(9):                         # Складываем антенны горизонтально
                line.append(data[i][j])                # Складываем антенны вертикально
            dataToAnalyze.append(line)

        '''
        for i in range(countPackets):
            for j in range(9):                         # Складываем антенны горизонтально
                dataToAnalyze.append(data[i][j])                
        '''
        dataToAnalyze = np.asarray(dataToAnalyze).T
        print("Размер набора извлеченных данных составляет " + str(dataToAnalyze.shape))
        return dataToAnalyze, countPackets


def extractData(dataToExtract, fromPosition, toPosition, step):
    startPosition = fromPosition
    dataToExtract = dataToExtract.T
    extractedData = []
    while startPosition<toPosition:
        extractedData.append(dataToExtract[startPosition : startPosition+step])
        startPosition = startPosition + step
    return np.asarray(extractedData)

def buildTrainDataSet(part1, part2, part3):
    trainX = []
    trainY = []
    max = np.min([len(part1), len(part2), len(part3)])
    for i in range(max):
        trainX.append(part1[i])
        trainX.append(part2[i])
        trainX.append(part3[i])
        trainY.append([1, 0, 0])
        trainY.append([0, 1, 0])
        trainY.append([0, 0, 1])
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)
    print("Размер массива X: " + str(trainX.shape))
    print("Размер массива Y: " + str(trainY.shape))
    return trainX,trainY


def visualizePCA3D(x):
    # import relevant libraries for 3d graph
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 10))

    # choose projection 3d for creating a 3d graph
    axis = fig.add_subplot(111, projection='3d')

    # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
    axis.scatter(x[:, 0], x[:, 1], x[:, 2])
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)
    print(x.shape)
    print("Я вошёл в визуалайз")
    fig.show()

def visualizePCA2D(x):
    plt.figure(figsize=(10, 10))
    plt.scatter(range(len(x[:, 1])), x[:, 1])
    #plt.scatter(x[:, 0], x[:, 2])
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.show()

def tryPCA():


    # import the breast _cancer dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    data.keys()

    # Check the output classes
    print(data['target_names'])

    # Check the input attributes
    print(data['feature_names'])
    # construct a dataframe using pandas
    df1 = pd.DataFrame(data['data'], columns=data['feature_names'])

    # Scale data before applying PCA
    scaling = StandardScaler()

    # Use fit and transform method
    scaling.fit(df1)
    Scaled_data = scaling.transform(df1)

    # Set the n_components=3
    principal = PCA(n_components=3)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)

    # Check the dimensions of data after PCA
    print(x.shape)
    # Check the values of eigen vectors
    # prodeced by principal components
    principal.components_
    plt.figure(figsize=(10, 10))
    plt.scatter(x[:, 0], x[:, 1], c=data['target'], cmap='plasma')
    plt.xlabel('pc1')
    plt.ylabel('pc2')

    # import relevant libraries for 3d graph
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 10))

    # choose projection 3d for creating a 3d graph
    axis = fig.add_subplot(111, projection='3d')

    # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
    axis.scatter(x[:, 0], x[:, 1], x[:, 2], c=data['target'], cmap='plasma')
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)

    # check how much variance is explained by each principal component
    print(principal.explained_variance_ratio_)


if __name__ == '__main__':
    # Работа с большим набором
    filename_full = "d://!!!CSI-Movement//My_25.11.2024//data_apml.json"
    dataset_full, countPackets = loadDataSet(filename_full)
    frameSize = 180


    # Для PCA необходима рамерность 2
    # Мы получаем размерность 3 (56х9хt)
    sub_x=dataset_full[:,0,:]
    print(dataset_full.shape)

    sub_x = np.zeros(shape=(9,56,countPackets))
    for i in range(9):
        sub_x[i] = dataset_full[:,i,:]
    print(sub_x.shape)

    voidDetectorKalmanPCA(sub_x[1])
    #voidDetectorKalmanPCA(sub_x[1])
    #voidDetector(sub_x[1])
    '''
    class1_full = extractData(dataset_full, 3100, 10000, frameSize)
    print("Размер первого класса:" + str(class1_full.shape))

    class2_full = extractData(dataset_full, 13300, 20000, frameSize)
    print("Размер второго класса:" + str(class2_full.shape))

    class3_full = extractData(dataset_full, 23000, 30200, frameSize)
    print("Размер третьего класса:" + str(class3_full.shape))


    # Работа с малым набором
    filename_min = "d://!!!CSI-Movement//My_25.11.2024//data_apml_min.json"
    dataset_min, countPackets = loadDataSet(filename_min)

    class1_min = extractData(dataset_min, 3100, 6300, frameSize)
    print("Размер первого класса:" + str(class1_min.shape))

    class2_min = extractData(dataset_min, 10000, 13300, frameSize)
    print("Размер второго класса:" + str(class2_min.shape))

    class3_min = extractData(dataset_min, 16100, 19200, frameSize)
    print("Размер третьего класса:" + str(class3_min.shape))

    testX, testY = buildTrainDataSet(class1_min, class2_min, class3_min)
    trainX, trainY = buildTrainDataSet(class1_full, class2_full, class3_full)


    # Формируем модель Conv1D

    model = Sequential()
    model.add(BatchNormalization())
    #model.add(GlobalMaxPooling1D())
    #model.add(Conv1D(filters=512, kernel_size=5, activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=9, activation='relu'))
    model.add(Dense(128, activation='relu'))
    #model.add(GlobalMaxPooling2D())                                                           #model.add(MaxPooling1D())GlobalAveragePooling1D
    model.add(GlobalAveragePooling2D())                                                        #model.add(MaxPooling1D())GlobalAveragePooling1D
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])    #loss = 'sparse_categorical_crossentropy'   categorical_crossentropy

    model.summary()

    st = time.time()
    start_time = time.time()

    #history = model.fit(trainX, trainY, epochs=10, batch_size=3, validation_data=(testX, testY), verbose=2)
    history = model.fit(trainX, trainY, epochs=10, batch_size=3, validation_split=0.2, verbose=2)

    print("--- cуммарно %s секунд ---" % (time.time() - st))


    print("Класс 1")
    result = model.predict(class1_min)
    class1 = 0
    class2 = 0
    class3 = 0
    for i in range(len(result)):
        class1 = class1+result[i][0]
        class2 = class2+result[i][1]
        class3 = class3+result[i][2]
    print(class1)
    print(class2)
    print(class3)
    print("Точность на классе " + str(class1/len(result)*100))


    print("Класс 2")
    result = model.predict(class2_min)
    class1 = 0
    class2 = 0
    class3 = 0
    for i in range(len(result)):
        class1 = class1 + result[i][0]
        class2 = class2 + result[i][1]
        class3 = class3 + result[i][2]
    print(class1)
    print(class2)
    print(class3)
    print("Точность на классе " + str(class2 / len(result)*100))

    print("Класс 3")
    result = model.predict(class3_min)
    class1 = 0
    class2 = 0
    class3 = 0
    for i in range(len(result)):
        class1 = class1 + result[i][0]
        class2 = class2 + result[i][1]
        class3 = class3 + result[i][2]
    print(class1)
    print(class2)
    print(class3)
    print("Точность на классе " + str(class3 / len(result)*100))

    '''


    '''
    plt.figure(figsize=(16, 8))
    plt.plot(dataset_full.T)
    markers = np.zeros(countPackets)
    markers[3100] = -100
    markers[10000] = -100
    markers[13300] = -100
    markers[20000] = -100
    markers[23000] = -100
    markers[30200] = -100
    plt.plot(markers)
    plt.show()
    
    


    
    plt.figure(figsize=(18, 9))
    plt.plot(dataset.T)
    markers = np.zeros(countPackets)
    markers[3100] = -100
    markers[6300] = -100
    markers[10000] = -100
    markers[13300] = -100
    markers[16100] = -100
    markers[19200] = -100
    plt.plot(markers)
    plt.show()
    '''



