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

xn = [1,1,1,1]; N = len(xn); print("xn=", xn)
nT = 1000
MXe = DTFT(xn,N,nT)
magMXe = np.abs(MXe)
phaMXe = np.angle(MXe,deg=True)

plt.figure(1)
w = np.linspace(0,2*np.pi,nT)
plt.subplot(2,1,1);plt.plot(w,magMXe,"b")
plt.grid();plt.ylabel("Magnitude of X(Omega)")
plt.title("Magnitude and Phase of X(Omega)")
plt.subplot(2,1,2); plt.plot(w,phaMXe,"g")
plt.grid();plt.ylabel("phase of X(Omega)")
plt.xlabel("frequency in radians")
plt.show()

Xk = DFT(xn,N);print("Xk=",Xk)
magXk = np.abs(Xk);print("Magnitude of Xk=",magXk)
phaXk = np.angle(Xk,deg=True);print("Phase of Xk=",phaXk)

plt.figure(2)
k = np.arange(N)
plt.subplot(2,1,1);plt.stem(k,magXk,"b")
plt.grid();plt.ylabel("Magnitude of X(k)")
plt.title("Magnitude and Phase of X(k)")
plt.subplot(2,1,2);plt.stem(k,phaXk,"g")
plt.grid();plt.ylabel("Phase of X(k)");plt.xlabel("k")
plt.show()

w = np.linspace(0,np.pi,nT)
nk1 = np.arange(5)
X1 = np.zeros(5); X1[0:4] = magXk
X2 = np.zeros(5); X2[0:4] = phaXk

plt.figure(3)
plt.subplot(2,1,1);
plt.plot(w*4/3.14, magMXe,"b:");plt.stem(nk1,X1,"magenta");
plt.grid();plt.ylabel("Magnitude ");plt.title("DTFT X(Omega) and DFT X(k)")
plt.subplot(2,1,2); plt.plot(w*4/3.14, phaMXe, "g:")
plt.stem(nk1, X2,"magenta");plt.grid();plt.ylabel("Phase");plt.xlabel("k")
plt.show()