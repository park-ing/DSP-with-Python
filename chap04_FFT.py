import numpy as np
import matplotlib.pylab as plt

N = 128 # 입력시퀀스 길이
n = np.arange(N)    # 순서축 설정
n1 = n[0:int(N/2)]
xn = np.cos(0.48*np.pi*n) + np.cos(0.52*np.pi*n)    # 입력이산신호

Xk = np.fft.fft(xn,N)   # FFT 스펙트럼

magXk = np.abs(Xk)  # 크기 스펙트럼
magXk1 = magXk[0:int(N/2)]

xnt = np.fft.ifft(Xk)   # 역FFT, 복원된 이산신호

plt.figure(1)
plt.subplot(3,1,1);plt.stem(n,xn,"blue")
plt.grid();plt.ylabel("Input x(n)")
plt.title("Input sequence x(n), FFT X(k), IFFT xr(n)")
plt.subplot(3,1,2); plt.stem(n1, magXk1,"green")
plt.grid();plt.ylabel("|X(k)|")
plt.subplot(3,1,3); plt.stem(n,xnt,"green")
plt.grid();plt.ylabel("Reconstructed xr(n)");plt.xlabel("k")
plt.show()