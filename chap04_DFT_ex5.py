import numpy as np
import matplotlib.pylab as plt
from chap04_DFT_IDFT import *

def DTFT(xn,N,T):   # 이산시간푸리에변환
    w = np.linspace(0,2*np.pi,T)    # 주파수축
    MXe = np.zeros(T, dtype = "complex64")  # DTFT 계수, 복소수
    for i in range(T):
        sum = 0 # DTFT 계수 계산
        MXe[i] = np.sin(2*w[i])/np.sin(w[i]/2)*np.exp(-1j*w[i]*3/2)
    return MXe  # DTFT 계수 리턴

# N = 4
xn = [1,1,1,1];N=4;print("xn=",xn,"N=",N)
Xk = DFT(xn,N)
magXk = np.abs(Xk);print("Magnitude of Xk=", magXk)

# N = 8
xn1 = [1,1,1,1,0,0,0,0]; N1=8;print("xn1=",xn1,"N1=",N1)
Xk1 = DFT(xn1, N1)
magXk1 = np.abs(Xk1);print("Magnitude of Xk1=", magXk1)

# N = 16
xn2 = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]; N2=16;print("xn2=",xn2,"N2=",N2)
Xk2 = DFT(xn2, N2)
magXk2 = np.abs(Xk2);print("Magnitude of Xk2=", magXk2)

nT = 1000
MXe = DTFT(xn, N, nT); magMXe = np.abs(MXe)
w = np.linspace(0,np.pi, nT)
nk1 = np.arange(5);nk2 = np.arange(9); nk3 = np.arange(17)
X1 = np.zeros(5); X1[0:4]=magXk
X2 = np.zeros(9); X2[0:8]=magXk1
X3 = np.zeros(17); X3[0:16]=magXk2

plt.subplot(3,1,1);
plt.plot(w*4/3.14,magMXe, "b:"); plt.stem(nk1,X1,"green");
plt.grid();plt.ylabel("X(k), N=4 ");plt.title("DTFT X(Omega) and DFT X(k)")
plt.subplot(3,1,2);plt.plot(w*8/3.14,magMXe,"b:");plt.stem(nk2,X2,"green");
plt.grid();plt.ylabel("X(k), N=8 ")
plt.subplot(3,1,3);plt.plot(w*16/3.14,magMXe,"b:");plt.stem(nk3,X3,"green");
plt.grid();plt.ylabel("X(k), N=16 "); plt.xlabel("k")
plt.show()

