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
imag = 0 + 1j

def aw(E):
    """
    - in the intervall [delta/2, delta/2+V0] is the 
      term in the square root always greater or equal to zero
    - the output will always be purely real or zero
    """
    #condition = E**2-(delta/2)**2
    # print(condition)
    #if condition >= 0:
    #    r = np.sqrt(condition)/a/t
    #else:
    #    r = i*np.sqrt(-condition)/a/t
    #return r
    return np.sqrt(E**2-(delta/2)**2)/a/t
        
def av(E, V0):
    """
    - in the intervall [delta/2, delta/2+V0 is the 
      term in the square root always lower or equal to zero]
    - the output will always be purely imaginary or zero
    """
    condition = E**2-(delta/2)**2-2*E*V0+V0**2
    #if condition >= 0:
    #    r = np.sqrt(condition)/a/t
    #else:
    #    r = imag*np.sqrt(-condition)/a/t
    return imag*np.sqrt(-condition)/a/t

def return_function(E, V0, r0):
    AV = av(E, V0)
    AW = aw(E)
    left = np.real(yv(0,-r0*AV))/jv(0,r0*AW)
    right = imag*AV/AW*(E+delta/2)/(V0-E-delta/2)*(np.imag(yv(1,-r0*AV)))/(jv(1,r0*AW))
    return left-right

V0 = 30e-3 #eV
r0 = 100 #Angström

x = np.linspace(delta/2, V0+delta/2, 1000000, endpoint=True)
x = x[1:-1]
y = []
for index, x_i in enumerate(x):
    print(int(index/len(x)*100))
    y.append(return_function(x_i, V0, r0))
y = np.array(y)
print(y)

plt.scatter((x-delta/2-V0)/V0,y)
plt.ylabel("Root function")
plt.xlabel("E/V0")
plt.grid(True)
plt.show()

