'''

Файл с функциями для генерации визальной информации для статьи

'''


import numpy as np
import matplotlib.pyplot as plt


def drawFirstGraph(data):
    numPackets = 3  # Количество паетов для графика с кадой пары антенн
    packetsForFirstGraph = np.zeros(shape=(9, 56, numPackets))
    for i in range(9):
        packetsForFirstGraph[i] = data[i, :, 0:numPackets]
    print("Размер болка в 100 пакетов с 1 антенны равен: " + str(packetsForFirstGraph.shape))

    plt.figure(figsize=(14, 7))
    plt.plot(packetsForFirstGraph[0], color='green')
    plt.plot(packetsForFirstGraph[1], color='red')
    plt.plot(packetsForFirstGraph[2], color='blue')
    plt.plot(packetsForFirstGraph[3], color='yellow')
    plt.plot(packetsForFirstGraph[4], color='purple')
    plt.plot(packetsForFirstGraph[5], color='black')
    plt.plot(packetsForFirstGraph[6], color='aqua')
    plt.plot(packetsForFirstGraph[7], color='brown')
    plt.plot(packetsForFirstGraph[8], color='grey')
    # for i in range(9):
    #    plt.plot(packetsForFirstGraph[i], label="Пакеты с "+str(i)+" антенны",  color='green')


    plt.xlabel('Поднесущие')

    plt.ylabel('Амплитуда, дб.')

    # plt.legend(["Пакеты с 1 антенны", "Пакеты со 2 антенны", "Пакеты с 3 антенны", "Пакеты с 4 антенны", "Пакеты с 5 антенны", "Пакеты с 6 антенны", "Пакеты с 7 антенны", "Пакеты с 8 антенны", "Пакеты с 9 антенны"])
    plt.show()
    plt.close()


def drawSecondGraph(data):
    plt.figure(figsize=(14, 7))

    plt.plot(data, color='red')

    plt.xlabel('Номер пакета')
    plt.ylabel('Амплитуда, дб.')
    plt.show()

def drawThirdGraph(data):
    plt.figure(figsize=(14, 7))

    plt.plot(data, color='red')



    plt.xlabel('Номер пакета')
    plt.ylabel('Амплитуда, дб.')
    plt.show()


