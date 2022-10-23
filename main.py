from scipy.special import jv, yv
from scipy.optimize import newton
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator



data = {"MoS2": {"band gap": 1.59,
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

TMD = data["MoS2"]
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
    condition = E**2-(delta/2)**2
    if condition >= 0:
        r = np.sqrt(condition)/a/t
    else:
        r = imag*np.sqrt(-condition)/a/t
    return r

def av(E, V0):
    condition = (E-V0)**2-(delta/2)**2
    if condition >= 0:
        r = np.sqrt(condition)/a/t
    else:
        r = imag*np.sqrt(-condition)/a/t
    return r


def return_function(E, V0, r0):
    AW = aw(E)
    AV = av(E, V0)
    left = np.real(yv(0,-r0*AV))/jv(0,r0*AW)
    right = imag*AV/AW*(E+delta/2)/(V0-E-delta/2)*np.imag(yv(1,-r0*AV))/(jv(1,r0*AW))
    if np.imag(left-right) != 0:
        print(f'AW={AW}, AV={AV}, imag*AV={imag*AV}, V0={V0}, r0={r0}')
        print('return func = ',left-right)
    return left-right

            
def simple_plot(V0=300e-3, r0=50, suppress_plot=True): #[V0]=eV, [r0]=Angström
    #V0 = 30e-3 #eV
    #r0 = 100 #Angström
    if not suppress_plot:
        x = np.linspace(delta/2, V0+delta/2, 10000, endpoint=True)
        x = x[1:-1]
        y = []
        for index, x_i in enumerate(x):
            y.append(return_function(x_i, V0, r0))
        y = np.array(y)
    bisect_bounds = np.linspace(delta/2+1e-15, V0+delta/2-1e-15, num=1000, endpoint=True)
    min_root = V0+delta/2
    for k in range(len(bisect_bounds)-1):
        a = bisect_bounds[k]
        b = bisect_bounds[k+1]
        if return_function(a, V0, r0)*return_function(b, V0, r0) > 0:
            continue
        elif return_function(a, V0, r0)*return_function(b, V0, r0)==0:
            root_min = np.min(np.abs([a,b]))
        else:
            root, conv = brentq(f=return_function, a=a, b=b, args=(V0,r0), full_output=True)
        if conv.converged and np.abs(return_function(root, V0=V0, r0=r0))<1e-9:
            min_root = root
            break
    EG = (min_root-delta/2)/V0
    if not suppress_plot:
        fig, ax = plt.subplots()
        ax.scatter((x-delta/2)/V0,y,color='blue')
        ax.axvline(EG, color="red", label=f'EG={EG}')
        ax.legend()
        ax.set_ylabel("Root function")
        ax.set_xlabel("E/V0")
        ax.set_title(f"V0={V0}, r0={r0}")
        ax.grid(True)
        plt.show()
    return min_root



def area_plot(V0_list, r0_list, suppress_plot=True):
    r0_list = np.array(r0_list)
    V0_list = np.array(V0_list)
    r0_X, V0_Y = np.meshgrid(r0_list, V0_list)
    EG = np.zeros(shape=r0_X.shape)
    for i in range(r0_X.shape[0]):
        for j in range(r0_X.shape[1]):
            print(f'{i}/{r0_X.shape[0]}, {j}/{r0_X.shape[1]}')
            V0 = V0_Y[i][j]
            r0 = r0_X[i][j]
            min_root = simple_plot(V0=V0, r0=r0, suppress_plot=suppress_plot)
            EG[i][j] = (min_root-delta/2)/V0
            print(f'V0={V0}, r0={r0}, EG = {EG[i][j]}')
    print(EG)
    
    cmap = plt.colormaps['bone']
    levels = MaxNLocator(nbins=11).tick_values(0,1.1)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    
    fig, ax = plt.subplots(1,1)
    im = ax.pcolormesh(a*t/delta/r0_X, V0_Y/delta, EG, shading='nearest', cmap=cmap, norm=norm)
    ax.set_ylabel(r'$\frac{V_0}{\Delta}$')
    ax.set_xlabel(r'$\frac{at}{\Delta r_0}$')
    fig.colorbar(im, ax=ax, label='uiae')
    fig.savefig("pcolormesh.pdf")
    
    fig, ax = plt.subplots(1,1)

    im = ax.contour(a*t/delta/r0_X, V0_Y/delta, EG, shading='nearest', levels=levels, cmap=cmap)
    im = ax.contourf(a*t/delta/r0_X, V0_Y/delta, EG, shading='nearest', levels=levels, cmap=cmap)
    ax.set_ylabel(r'$\frac{V_0}{\Delta}$')
    ax.set_xlabel(r'$\frac{at}{\Delta r_0}$')
    fig.colorbar(im, ax=ax, label='uiae')
    fig.savefig("contourf.pdf")
                    

if __name__ == '__main__':
    #simple_plot(V0=500e-3, r0=10)
    r0_ax = np.linspace(0,0.6,100, endpoint=True)
    r0_ax = r0_ax[1:]
    V0_ax = np.linspace(0,0.25,100, endpoint=True)
    V0_ax = V0_ax[1:]
    area_plot(V0_list = V0_ax*delta, r0_list=a*t/delta/r0_ax, suppress_plot=True)