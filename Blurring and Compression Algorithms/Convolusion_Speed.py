from imageio import imread
from scipy import signal
import time
import numpy as np
import matplotlib.pyplot as plt

filename = 'cow.png'
f = imread(filename,as_gray=True)
plt.imshow(f,cmap='gray')

def middel_verdi_filter(dim):
    filter = np.zeros((dim,dim))
    antall_bildet_elementer = dim*dim
    for i in range(0,dim):
        for j in range(0,dim):
            filter[i][j] = 1/antall_bildet_elementer
    return filter

def convolve(bildet):
    #middelverdi filtering ved romlig konvolusjon
    middelverdifilter = middel_verdi_filter(15)
    bildet_romlig = signal.convolve2d(bildet,middelverdifilter,'same')
    # middelverdi filtering ved Fourier frekvense domenet

    #paddingen av filteret
    x,y = middelverdifilter.shape
    filter_fourier = np.zeros((bildet.shape))
    filter_fourier [:x,:y] = middelverdifilter
    #filtrering ved frikvens domenet
    filter_fourier = np.fft.fft2(filter_fourier)
    bildet_fourier = np.fft.fft2(bildet)
    bildet_fourier = np.fft.ifft2(bildet_fourier*filter_fourier)
    bildet_fourier = np.real(bildet_fourier)

    return bildet_fourier , bildet_romlig

def generell_konvolusjon(bildet,filter_dimensjonen):
    romlig_tid_start = time.perf_counter()
    middelverdifilter = middel_verdi_filter(filter_dimensjonen)
    Romlig_konvolusjon = signal.convolve2d(bildet,middelverdifilter)
    romlig_tid_slutt = time.perf_counter()
    romlig_kjoretid = romlig_tid_slutt - romlig_tid_start
    x,y = middelverdifilter.shape
    filter_fourier = np.zeros((bildet.shape))
    filter_fourier [:x,:y] = middelverdifilter

    fourier_tid_start = time.perf_counter()
    filter_fourier = np.fft.fft2(filter_fourier)
    bildet_fourier = np.fft.fft2(bildet)
    bildet_fourier = np.fft.ifft2(bildet_fourier*filter_fourier)
    bildet_fourier = np.real(bildet_fourier)
    fourier_tid_slutt = time.perf_counter()
    fourier_kjoretid = fourier_tid_slutt-fourier_tid_start

    return fourier_kjoretid,romlig_kjoretid
##############################################################################

#kjøreeksemplene
plt.imshow(convolve(f)[0],cmap = 'gray')
plt.show()

plt.imshow(convolve(f)[1],cmap = 'gray')
plt.show()

def plot(bildet):
    romlig = []
    fourier = []
    for i in range(2,50,1):
        fourier_tiden = generell_konvolusjon(bildet,i)[0]
        romlig_tiden =generell_konvolusjon(bildet,i)[1]
        plot_fourier = fourier.append(fourier_tiden)
        plot_romlig = romlig.append(romlig_tiden)
    filter_storelsen = np.linspace(2,50,48)
    plt.figure()
    plt.plot(filter_storelsen,romlig,'r',label = 'romlig')
    plt.plot(filter_storelsen,fourier,'g',label = 'Fourier')
    plt.xlabel('filter størelsen')
    plt.ylabel('tiden i sekunder')
    plt.legend()
    plt.show()

plot(f)
