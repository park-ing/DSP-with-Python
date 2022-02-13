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

N = 10
xn = np.zeros(N,dtype="float")
for i in range(N):
    xn[i]=np.cos(0.48*np.pi*i) + np.cos(0.52*np.pi*i)
print("x(n)=",xn);print()
Xk = DFT(xn,N)
magXk = np.abs(Xk); print("Magnitude of X(k)=",magXk)
n = np.arange(N)
fn = np.linspace(0,np.pi,int(N/2)+1);print("fn=",fn)
plt.figure(1)
plt.subplot(2,1,1);plt.stem(n,xn,"blue")
plt.grid();plt.ylabel("x(n)")
plt.title("Sequence x(n) and DFT X(k), N=10")
plt.subplot(2,1,2);plt.stem(fn,magXk[0:int(N/2)+1],"green")
plt.grid();plt.ylabel("|X(k)|");plt.xlabel("frequency in radian")
#plt.show()

N1 = 100
xn1 = np.zeros(N1,dtype="float")
xn1[0:N]=xn
Xk1 = DFT(xn1,N1)
magXk1 = np.abs(Xk1)

n1 = np.arange(N1)
fn = np.linspace(0,np.pi,int(N1/2)+1)
plt.figure(2)
plt.subplot(2,1,1);plt.stem(n1,xn1,"blue")
plt.grid();plt.ylabel("x(n)")
plt.title("Sequence x(n) and DFT X(k), N=100, 90 zero-padding")
plt.subplot(2,1,2);plt.stem(fn,magXk1[0:int(N1/2)+1],"green")
plt.grid();plt.ylabel("|X(k)|");plt.xlabel("frequency in radian")
#plt.show()

N2 = 100
xn = np.zeros(N2,dtype="float")
for i in range(N2):
    xn[i] = np.cos(0.48*np.pi*i) + np.cos(0.52*np.pi*i)
Xk2 = DFT(xn,N2)
magXk2 = np.abs(Xk2)

n2 = np.arange(N2)
fn = np.linspace(0,np.pi,int(N2/2)+1)
plt.figure(3)
plt.subplot(2,1,1);plt.stem(n2,xn,"blue")
plt.grid();plt.ylabel("x(n)")
plt.title("Sequence x(n) and DFT X(k), N=100, no zero-padding")
plt.subplot(2,1,2);plt.stem(fn,magXk2[0:int(N1/2)+1],"green")
plt.grid();plt.ylabel("|X(k)|");plt.xlabel("frequency in radian")
plt.show()
