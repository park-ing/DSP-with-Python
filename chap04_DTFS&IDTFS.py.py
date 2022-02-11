import numpy as np
import matplotlib.pylab as plt

def DTFS(xn, N):    # 이산주기신호와 주기로부터 DTFS계수 구하는 함수
    WN = np.exp(-1j*(2*np.pi/N))    # 복소지수항
    Xk = np.zeros(N, dtype="complex64") # DTFS 계수 array 설정
    for i in range(N):  # i는 원래 식에서 k 역할
        sum = 0
        for j in range(N):
            sum = sum + xn[j]*np.power(WN, i*j) # 개별 DTFS 계수 계산
        Xk[i] = sum
    return Xk

xn = [0,1,2,3]; print("x(n)=", xn)  # 이산주기신호의 기본주기구간
N = len(xn) # 이산주기신호의 주기
Xk = DTFS(xn, N); print("X(k)=",Xk);print("|X(k)|=", np.abs(Xk))    # DTFS 계수

n = np.arange(N)    # 순서시퀀스 생성
plt.subplot(2,1,1); plt.stem(n,xn,"blue");plt.ylabel("x(n)")    # 이산주기신호
plt.grid(); plt.title("DTFS of a discrete periodic signal x(n)")
plt.subplot(2,1,2); plt.stem(n, np.abs(Xk), "green")    # DTFS 계수
plt.xlabel("k");plt.ylabel("|X(K)|")
plt.grid();plt.show()

