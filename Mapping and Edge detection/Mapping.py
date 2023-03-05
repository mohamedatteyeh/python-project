from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from math import cos , sin , pi ,radians,sqrt,floor,ceil,exp,atan
import numba

filename = 'portrett.png'
f =imread(filename,as_gray=True)
plt.imshow(f,cmap='gray')

filename1 = 'geometrimaske.png'
f1 = imread(filename1,as_gray=True)
plt.imshow(f1,cmap = 'gray')


#oppgave 1

@numba.jit
def oppgave1(image):
    yn = np.zeros((170,291))
    x = []
    x1= []
    xd = []
    xd1 = []
    s_en = 64
    mu_n = 127
    n,m = f.shape
    for i in range(n):
        sx = sum(image[:][i])
        x.append(sx)
    mu = sum(x)/(n*m)
    for i in range(n):
        fn = (image-mu)**2
        sxn = sum(fn[:][i])
        xd.append(sxn)
    s_e = sqrt(sum(xd)/(n*m))
    a = s_en/s_e
    b = mu_n - a*mu
    for i in range(n):
        for j in range(m):
            yn[i][j] = a*image[i][j] + b
# Checking if the standard error and the mean is equal to 64 and 127
    for i in range(n):
        sx1 = sum(yn[:][i])
        x1.append(sx1)
    mu1 = sum(x1)/(n*m)
    for i in range(n):
        fn1 = (yn-mu1)**2
        sxn1 = sum(fn1[:][i])
        xd1.append(sxn1)
    s_e1 = sqrt(sum(xd1)/(n*m))
# making the picture in the limits of an 8 bit:
    mx = np.max(yn)
    mn = np.min(yn)

    yn1 = 255/(mx-mn) * (yn-mn)
    return yn1,s_e1,mu1

@numba.jit
def Transformation_matrice():
    x = [[119.2,84,131],[68.4,89,108],[1,1,1]]
    y = [[342,168,257],[262,257,440],[1,1,1]]
    T = np.dot(y,np.linalg.inv(x))
    return T

@numba.jit
def forlengs_mapping(innbildet,utbildet,T):
    m,n = utbildet.shape
    M,N = innbildet.shape
    overlapping = np.zeros(utbildet.shape)
    for i in range(N):
        for j in range(M):
            p=[[i],[j],[1]]
            p1=np.dot(T,p)
            y,x = p1[0],p1[1]
            if (0 <= x < m) & (0 <= y < n):
                overlapping[int(x)][int(y)] = innbildet[j][i]


    return overlapping

@numba.jit
def baklengs_mapping(innbildet,utbildet,T):

    m,n = utbildet.shape
    nabo = np.zeros(utbildet.shape)
    bilinear = np.zeros(utbildet.shape)
    for i in range(n):
        for j in range(m):
            p = [[i],[j],[1]]
            p1 = np.dot(np.linalg.inv(T),p)
            y,x = p1[0],p1[1]
            if (0<=x<n)&(0<=y<m):
                nabo [j][i] = innbildet [int(x)][int(y)]


            x0,y0 = int(floor(x)),int(floor(y))
            x1,y1 = int(ceil(x)),int(ceil(y))
            dx = x - x0
            dy = y - y0
            p = innbildet[x0][y0] + (innbildet[x1][y0] - innbildet[x0][y0])*dx
            q = innbildet[x0][y1] + (innbildet[x1][y1] - innbildet[x0][y1])*dx
            bilinear [j][i] = p + (q-p) *dy

    return nabo , bilinear

#kjøre eksamplet

k = forlengs_mapping(oppgave1(f)[0],f1,Transformation_matrice())
k1 = baklengs_mapping(oppgave1(f)[0],f1,Transformation_matrice())[0] #nabo
k2 = baklengs_mapping(oppgave1(f)[0],f1,Transformation_matrice())[1] #bilinear

plt.imshow(k,cmap='gray')
plt.show()

plt.imshow(k1,cmap='gray')
plt.show()


plt.imshow(k2,cmap='gray')
plt.show()


print('Det nye standardavviket %g og den nye middel verdien %g' %(oppgave1(f)[1],oppgave1(f)[2]))
