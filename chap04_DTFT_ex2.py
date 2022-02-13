import numpy as np
import matplotlib.pylab as plt

N = 500
n = np.arange(30) # array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,....])
xn = np.power(0.5, n)
Omega = np.arange(-2*N,2*N)*2*np.pi/N   # 오메가 값들을 많이 얻기 위하여
print(Omega)
X = np.exp(1j*Omega)/(np.exp(1j*Omega)-0.5)
magX = np.abs(X)

plt.subplot(2,1,1); plt.stem(n,xn,"b"); plt.grid()
plt.ylabel("x(n)");plt.title("x(n) & X(Omega), DTFT of x(n)")
plt.subplot(2,1,2); plt.plot(Omega, magX, "b")
plt.xlabel("Omega, frequency in radians"); plt.ylabel("|X(Omega)|");
plt.xlim(-4*np.pi,4*np.pi);
plt.grid()
plt.show()
