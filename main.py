from scipy.special import jv, yv
import numpy as np
import matplotlib.pyplot as plt


data = {"MoSe2": {"band gap": 1.59,
                  "a": 3.19,
                  "t": 1.059,
                  "Ev": -5.86,
                  "Ec": -4.27,
                  "g1": 0.055,
                  "g2": 0.196,
                  "g3": -0.123},
        "WS2": {"band gap": 1.54,
                          "a": 3.19,
                          "t": 1.075,
                          "Ev": -5.50,
                          "Ec": -3.93,
                          "g1": -0.288,
                          "g2": -0.639,
                          "g3": 0.105}}

TMD = data["MoSe2"]
delta = TMD["band gap"]
a = TMD["a"]
t = TMD["t"]
Ev = TMD["Ev"]
Ec = TMD["Ec"]
g1 = TMD["g1"]
g2 = TMD["g2"]
g3 = TMD["g3"]
i = 0 + 1j

def aw(E):
    condition = E**2-(delta/2)**2
    print(condition)
    if condition >= 0:
        r = np.sqrt(condition)/a/t
    else:
        r = i*np.sqrt(-condition)/a/t
    return r
        
def av(E, V0):
    condition = E**2-(delta/2)**2-2*E*V0+V0**2
    print(condition)
    if condition >= 0:
        r = np.sqrt(condition)/a/t
    else:
        r = i*np.sqrt(-condition)/a/t
    return r

def return_function(E, V0, r0):
    AV = av(E, V0)
    AW = aw(E)
    left = np.real(yv(0,r0*AV))/jv(0,r0*AW)
    right = i*AV/AW*(np.imag(yv(1,-1*r0*AV)))/(jv(1,r0*AW))
    return left-right

V0 = 50e-3 #eV
r0 = 500 #Angström

x = np.linspace(0, 2*V0, 10000, endpoint=True)
y = []
for i in x:
    y.append(return_function(i, V0, r0))
y = np.array(y)

plt.plot(x/V0,y)
plt.ylabel("Root function")
plt.xlabel("E/V0")
plt.grid(True)
plt.show()

