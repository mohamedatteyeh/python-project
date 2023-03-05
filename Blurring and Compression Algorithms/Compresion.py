from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import numba
from math import log
image = imread('uio.png', as_gray = True)

Q = [[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]

def c(a):
    if a == 0:
        return 1/np.sqrt(2)
    else:
        return 1
@numba.jit
def DCT_forward(f):
    F = np.zeros((8,8))
    for u in range(8):
        for v in range(8):
            a = 0.25*c(u)*c(v)
            b = 0
            for x in range(8):
                for y in range(8):
                    b += f[x][y] * np.cos((2*x+1)*u*np.pi/16)*np.cos((2*y+1)*v*np.pi/16)

            F[u][v]= a * b
    return(F)
@numba.jit
def DCT_backwards(F):
    f = np.zeros((8,8))
    for x in range(8):
        for y in range(8):
            b=0
            for u in range(8):
                for v in range(8):
                    b += c(u)*c(v)* F[u][v]* np.cos((2*x+1)*u*np.pi/16)*np.cos((2*y+1)*v*np.pi/16)

            f[x][y] = b/4
    return (f)
@numba.jit
def rekonstruksjonen(bildet,q,Q):
    x,y = bildet.shape
    x_,y_ = int(x/8),int(y/8)
    reconstructed_blokk = np.zeros((x_,y_),dtype = np.ndarray)
    transformert_blokk = np.zeros((x_,y_),dtype=np.ndarray)
    entropi_measuring = np.zeros(bildet.shape)
    reconstructed_image = np.zeros(bildet.shape)
    for i in range(0,x,8):
        for j in range(0,y,8):
            transformert_blokk[int(i/8)][int(j/8)] = DCT_forward(bildet[i:i+8,j:j+8])
            transformert_blokk[int(i/8)][int(j/8)] = np.round(np.divide(transformert_blokk[int(i/8)][int(j/8)] , np.multiply(q,Q)))
    for i in range(0,x,8):
        for j in range(0,y,8):
            entropi_measuring[i:i+8,j:j+8] = transformert_blokk[int(i/8)][int(j/8)]
    for i in range(x_):
        for j in range(y_):
            transformert_blokk[i][j] =  np.round(np.multiply(transformert_blokk[i][j],np.multiply(q,Q)))
            reconstructed_blokk[i][j] = DCT_backwards(transformert_blokk[i][j])
    reconstructed_blokk += 128
    for i in range(0,x,8):
        for j in range(0,y,8):
            reconstructed_image[i:i+8,j:j+8] = reconstructed_blokk[int(i/8)][int(j/8)]
    return(reconstructed_image,entropi_measuring)

def entropi(bildet):
    bildet = bildet.flatten()
    N = len(bildet)
    n = np.histogram(bildet,bins = 256)[0]
    p = np.divide(n,N)
    H = 0
    for i in range(256):
        if p[i] > 1e-11:

            H += p[i]*log(p[i],2)

    return (-H)



############################################################################
#kjøre eksamplet for rekonstruksjonene
q = [0.1,0.5,2,8,32]
for i in q:
    print(i,entropi(image))
    print(i,entropi(rekonstruksjonen(image,i,Q)[1])) # Entropi for første filtret
    print(i,entropi(rekonstruksjonen(image,i,Q)[0])) # Entropi for andre filtret

plt.imshow(rekonstruksjonen(image,0.1,Q),cmap='gray')
plt.show()

plt.imshow(rekonstruksjonen(image,0.5,Q),cmap='gray')
plt.show()

plt.imshow(rekonstruksjonen(image,2,Q),cmap='gray')
plt.show()

plt.imshow(rekonstruksjonen(image,8,Q),cmap='gray')
plt.show()

plt.imshow(rekonstruksjonen(image,32,Q),cmap='gray')
plt.show()

















##
