'''

Файл с функциями для фильтрациии фильтрами Баттерворта

'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
def lowpassButter(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    # Code used to display the result
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    ax1.plot(data)
    ax1.set_title("Original Signal")
    ax1.margins(0, .1)
    ax1.grid(alpha=.5, ls='--')
    ax2.plot(filtered_data)
    ax2.set_title("Low-Pass Filter (50 Hz)")
    ax2.grid(alpha=.5, ls='--')
    plt.tight_layout()
    plt.show()
    '''
    return filtered_data

def highpassButter(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    # Code used to display the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    ax1.plot(data)
    ax1.set_title("Original Signal")
    ax1.margins(0, .1)
    ax1.grid(alpha=.5, ls='--')
    ax2.plot(filtered_data)
    ax2.set_title("High-Pass Filter (20 Hz)")
    ax2.grid(alpha=.5, ls='--')
    plt.tight_layout()
    plt.show()
    return filtered_data

def bandpassButter(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    # Code used to display the result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    ax1.plot(data)
    ax1.set_title("Original Signal")
    ax1.margins(0, .1)
    ax1.grid(alpha=.5, ls='--')
    ax2.plot(filtered_data)
    ax2.set_title("Band-Pass Filter (10-50 Hz)")
    ax2.grid(alpha=.5, ls='--')
    plt.tight_layout()
    plt.show()
    return filtered_data


def lowPassCutoffFrequencyButter(data: np.ndarray, sample_rate: float):
    # Load sample data from a WAV file

    times = np.arange(len(data)) / sample_rate

    # Plot the original signal
    plt.plot(times, data, '.-', alpha=.5, label="original signal")

    # Plot the signal low-pass filtered using different cutoffs
    for cutoff in [20,15,12,6,4,2]:
        sos = scipy.signal.butter(5, cutoff, 'lowpass', fs=sample_rate, output='sos')
        filtered = scipy.signal.sosfiltfilt(sos, data)
        plt.plot(times, filtered, label=f"low-pass {cutoff} Hz")

    plt.legend()
    plt.grid(alpha=.5, ls='--')
    #plt.axis([0.35, 0.5, None, None])
    plt.show()
