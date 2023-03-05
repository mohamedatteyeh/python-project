from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import cos , sin , pi ,radians,sqrt,floor,ceil,exp,atan
import numba

filename2 = 'cellekjerner.png'
f2 = imread(filename2,as_gray=True)
plt.imshow(f2,cmap = 'gray')


def gauss(sigma):
    n = np.round(1+8*sigma)

    gaussf = np.zeros((n,n))

    l = int((n-1)/2)

    A = 1/(2. * pi * (sigma**2))

    for i in range(-l,l+1):
        for j in range(-l,l+1):
            h = exp(-(i**2 + j**2)/(2.*sigma**2))
            gaussf[i+l][j+l] = h

    g = gaussf * A

    return g

def konvolusjon(innbildet,filter):
    M,N = innbildet.shape
    m,n = filter.shape
    filter = np.rot90(filter,2)
    a = int(m-1/2)
    b = int(n-1/2)
    utbildet = np.zeros(innbildet.shape)

    # Padding med nærmeste piksel

    padding = np.zeros((M+a,N+b))

    padding [int(a/2):int(-a/2), int(b/2):int(-b/2)] = innbildet

    p_oppe =  innbildet[0:int(a/2),:]
    padding [0:int(a/2),int(b/2):int(-b/2)] = np.flip(p_oppe,0)

    p_venstre =  innbildet[:,0:int(b/2)]
    padding [int(a/2):int(-a/2),0:int(b/2)] = np.flip(p_venstre,1)

    p_nede =  innbildet[int(-a/2):,:]
    padding [int(-a/2):,int(b/2):int(-b/2)] = np.flip(p_nede,0)

    p_hoyere =  innbildet[:,int(-b/2):]
    padding [int(a/2):int(-a/2),int(-b/2):] = np.flip(p_hoyere,1)

    p_kant1 =  innbildet[M-1][N-1]
    padding [0:int(a/2),0:int(b/2)] = p_kant1

    p_kant2 =  innbildet[-M][-N]
    padding [int(-a/2):,int(-b/2):] = p_kant1

    p_kant3 =  innbildet[M-a][N-b]
    padding [int(-a/2):,0:int(b/2)] = p_kant1

    p_kant4 =  innbildet[int(a/2)][-N]
    padding [0:int(a/2),int(-b/2):] = p_kant1

    for i in range(M):
        for j in range(N):
            utbildet[i,j] = np.sum(padding[i:(i+n),j:(j+m)]*filter)

    return utbildet

def gradient(innbildet):

    hx = np.array(([0,1,0],[0,0,0],[0,-1,0]))
    hy = np.array(([0,0,0],[1,0,-1],[0,0,0]))

    gx = konvolusjon(innbildet,hx)
    gy = konvolusjon(innbildet,hy)

    M = np.sqrt(gx**2 + gy**2)
    Theta = np.arctan(gy/gx)
    Theta = Theta * (180/pi)

    return M,Theta

def NMS (m1,theta):
    n,m = m1.shape
    kant= np.zeros(m1.shape)
    for i in range(1,n-1):
        for j in range(1,m-1):
            retningen = theta[i,j]
            if (0 <= retningen < 22.5) or (157.5 <= retningen <= 180):
                a = m1[i,j+1]
                b = m1[i,j-1]
            elif (22.5 <= retningen < 67.5):
                a = m1[i-1,j-1]
                b = m1[i+1,j+1]
            elif (67.5 <= retningen < 112.5):
                a = m1[i-1,j]
                b = m1[i+1,j]
            else:
                a = m1[i+1,j-1]
                b = m1[i-1,j+1]

            if (m1[i,j] >= a) and (m1[i,j] >= b):
                kant[i,j] = m1[i,j]


    return kant

def hysterese(tl,th,innbildet_tynet):
    m,n = innbildet_tynet.shape
    w = 255
    s = 255
    wx,wy = np.where((innbildet_tynet >= tl) & (innbildet_tynet <= th))
    sx,sy = np.where(innbildet_tynet >= th)
    zx,zy = np.where(innbildet_tynet < tl)
    innbildet_tynet[wx,wy] = w
    innbildet_tynet[sx,sy] = s
    for i in range(1,m-1):
        for j in range(1,n-1):
            if(innbildet_tynet[i,j] == w):
                if ((innbildet_tynet[i+1, j-1] == s) or (innbildet_tynet[i+1, j] == s) or (innbildet_tynet[i+1, j+1] == s) or (innbildet_tynet[i, j-1]==s) or (innbildet_tynet[i, j+1]==s) or (innbildet_tynet[i-1, j-1] == s ) or (innbildet_tynet[i-1, j] == s ) or (innbildet_tynet[i-1, j+1] == s)):
                    innbildet_tynet[i,j] = s
                else:
                    innbildet_tynet[i,j] = 0



    return innbildet_tynet


#kjøreeksamplet
sigma = 5
Tl = 9
Th = 14
filtrert_bildet= konvolusjon(f2,gauss(sigma))
M,theta = gradient(filtrert_bildet)
kant = NMS(M,theta)
Teskelen = hysterese(Tl,Th,kant)
plt.imshow(Teskelen, cmap = 'gray')
plt.show()
