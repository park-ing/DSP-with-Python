import numpy as np

def DFT(xn, N): # 이산푸리에변환
    WN = np.exp(-1j*(2*np.pi/N))  # 회전인자 계산
    Xk = np.zeros(N, dtype="complex64")   # DFT 계수 어레이 설정
    for i in range(N):
        sum = 0
        for j in range(N):
            sum = sum + xn[j]*np.power(WN,i*j)  # DFT 계수 계산
        Xk[i] = sum
    return Xk   # DFT 계수 리턴

def IDFT(Xk, N): # 역 이산푸리에변환
    WN = np.exp(-1j*(2*np.pi/N))  # 회전인자 계산
    xn = np.zeros(N, dtype="complex64")   # 복원 이산신호 어레이 설정
    for i in range(N):
        sum = 0
        for j in range(N):
            sum = sum + Xk[j]*np.power(WN,-i*j)  # 역 DFT 계산
        xn[i] = sum/N
    return xn   # IDFT 계수 리턴    

xn = [1,1,1,1];print("xn=",xn)
N = 4
Xk = DFT(xn,N);print("Xk=",Xk)
magXk = np.abs(Xk); print("Magnitude of Xk=",magXk)
phaXk = np.angle(Xk)*180/np.pi
print("Phase of Xk=", phaXk)
xn = np.real(IDFT(Xk,N))    # real값 
xn = IDFT(Xk,N)
print("Reconstructed xn=",xn)