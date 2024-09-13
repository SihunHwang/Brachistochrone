import numpy as np
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
import scipy as sp
from scipy import stats
import sympy as smp
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.pylab import subplot
import statistics

G = 6.6743e-11
M = 5.9722e24
R = 6.378e6

matplotlib.rc('font', size=17)

def main():
    data = load_data('PREM_1s.csv')
    
    '''# plot density(r)
    plt.plot(data['radius'].to_list(), data['density'].to_list())
    plt.xlabel('radius $(m)$')
    plt.ylabel('density $(kg/m^3)$')
    plt.show()'''
    
    
    ''' #integral of r * density(r) dr
    PREM_rdensity = data['density']*data['radius']
    PREM_integral_rdensity = cumulative_trapezoid(PREM_rdensity, data['radius'], initial=0)
    # I(r)
    PREM_integral = (4/3)*np.pi* (PREM_integral_rdensity[-1] - PREM_integral_rdensity)

    # index of point of change = 39 = print(data['radius'].tolist().index(3480000))
    PREM_integral_quadratic = PREM_integral[:39+1]
    PREM_integral_linear = PREM_integral[39+1:]
    
    # curve fit
    def f_linear(x, m, c):
        return m*x +c
    
    m0 = - PREM_integral_linear[0]/(R - data['radius'][39+1])
    c0 =  - m0*R
    initial_guess = [m0, c0]
    
    p_linear, cov = curve_fit(f_linear, data['radius'][39+1:], PREM_integral_linear, p0=initial_guess) 
    def f_quadratic(x, a, b):
        return a*(x**2) + b
    
    #p_quadratic, cov = curve_fit(f_quadratic, data['radius'][:39+1], PREM_integral_quadratic, p0=[1e6,5.56e17])
    p_linear = [-9.61536360e+10, 6.11076318e+17] # m, c
    p_quadratic = [-2.39244698e+04, 5.54404470e+17] # a, b
    def I(r):
        if (r < 3480000):
            return f_quadratic(r, p_quadratic[0], p_quadratic[1])
        else:
            return f_linear(r, p_linear[0], p_linear[1])'''
    
    
    '''plt.plot(data['radius'].to_list(),[I(r) for r in data['radius']] , label='integral_fitted', color='blue')
    plt.scatter(data['radius'].to_list(),PREM_integral , label='PREM_integral', color='red', s=6)
    plt.xlabel('radius $(m)$')
    plt.ylabel('$I(r)$')
    plt.legend()
    plt.show()'''
    
    ''' # data for 3D graph
    def drdt(r, r0):
        return (r/r0)*np.sqrt(((r**2)* I(r0) - (r0**2)* I(r)) / I(r))
    def dtdr(r, r0):
        return 1 / drdt(r, r0)
    
    def time_integrand(r, r0):
        f = (drdt(r, r0)**2 + r**2) / (2*G*I(r))
        return 2*np.sqrt(f)/drdt(r, r0)'''
    
    '''r = np.linspace(1000, R, 50)
    r0 = np.linspace(1000, R, 40)
    #T = [quad(time_integrand, r0i + 1000, R-1000, args=(r0i))[0] for r0i in r0]
    time_integral = []
    for i in range(len(r0)):
        x = np.linspace(r0[i] + 1000 , R - 1000, 1000)
        y = [time_integrand(ri, r0[i]) for ri in x]
        time_integral.append(np.nanmax(cumulative_trapezoid(y,x, initial=0)))
    
    t = [[quad(dtdr, r0i, ri, args=(r0i))[0] for ri in r] for r0i in r0]'''

    ''' # 3D figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Labeling the X Axis
    ax.set_xlabel(r'$\theta$')
    # Labeling the Y-Axis
    ax.set_ylabel(r'$r_0$')
    # Labeling the Z-Axis
    ax.set_zlabel(r'$r$')
    for i in range(len(r0)):
        ax.plot(t[i], [r0[i]]*len(r), r)
    plt.show()'''

    #tAB = [np.nanmax(ti) * 2 for ti in t]
    '''plt.scatter(tAB, r0, label=r'Numerical Data; Non-uniform $\rho$', s=1, color='red')
    plt.plot([(R-r0i)*np.pi/R for r0i in r0],r0, label=r'$r_0 = R(1 - \theta_{AB} / \pi)$', color='blue')
    plt.xlabel(r'$\theta_{AB}$')
    plt.ylabel(r'$r_0$')
    plt.legend()
    plt.show()'''

    '''fig = plt.figure()
    axc = fig.add_subplot()
    
    ax = fig.add_subplot(111, polar=True)

    # the following 4 commands align the two axes 
    axc.set_aspect('equal')
    axc.set_xlim(-R,R)
    axc.set_ylim(-R,R)
    ax.set_rlim(0,R)

    for i in range(len(r0)):
        a = (R * tAB[i]) / (2 * np.pi)
        theta = np.linspace(0, tAB[i], 100)
        x = ((R-a)*np.cos(theta)) + (a*np.cos(theta*((R-a)/a)))
        y = ((R-a)*np.sin(theta)) - (a*np.sin(theta*((R-a)/a)))

        u = x*np.cos(-tAB[i]/2) - y*np.sin(-tAB[i]/2)
        v = x*np.sin(-tAB[i]/2) + y*np.cos(-tAB[i]/2)

        ax.plot([-1 * j for j in t[i]] +[0]+ t[i], r.tolist() +[r0[i]]+ r.tolist(), color='green')
        axc.plot(u, v, color='orange')
    ax.plot(np.linspace(0, 2*np.pi, 100), [R]*100, color='black')
    ax.grid(False)
    ax.set_axis_off()
    axc.set_axis_off()
    plt.show()'''
    
    '''plt.scatter(tAB, time_integral, color='orange', label='PREM')
    plt.plot(tAB, [np.pi*np.sqrt((1 - (1 - tABi/np.pi)**2) / (G*M/(R**3))) for tABi in tAB], color='blue', label=r'Uniform $\rho$')
    plt.legend()
    plt.xlabel(r'$\theta_{AB}$')
    plt.ylabel(r'Time $(s)$')
    plt.show()'''

    '''difference = np.array(time_integral)[:-1] - np.array([np.pi*np.sqrt((1 - (1 - tABi/np.pi)**2) / (G*M/(R**3))) for tABi in tAB])[:-1]
    std = np.std(difference)
    print(difference)
    print(std)'''
    thetaAB = np.linspace(0,np.pi, 100)

    x0 = R * np.sin(thetaAB/2)
    y0 = R * np.cos(thetaAB/2)
    w_l = np.sqrt(G*M/(R**3))
    T_l = np.pi/w_l
    J_l = (G*M/(R**3))**2 * ((x0**2)/2 + y0**2)

    a = R*thetaAB / (2*np.pi)
    w_h = np.sqrt((G*M/(R**3)) / (1 - (1 - thetaAB/np.pi)**2)) * thetaAB / np.pi
    T_h = thetaAB/w_h
    J_h = (G*M/(R**3))**2 * (T_h * (a**2 + (R-a)**2) - (2*(R-a)*(a**2)/R*w_h)*np.sin(R*w_h*T_h/a))
    J_h[0] = 0
    
    plt.plot(thetaAB, J_l, label='J_l')
    plt.plot(thetaAB, J_h, label='J_h')
    plt.xlabel(r'$\theta_{AB}$')
    plt.legend()
    plt.show()

    plt.plot(thetaAB, J_l, label='J_l')
    plt.xlabel(r'$\theta_{AB}$')
    plt.legend()
    plt.show()
    


def load_data(path):
    data = pd.read_csv(path, names=('radius','depth','density','Vpv','Vph','Vsv','Vsh','eta','Q-mu','Q-kappa'))
    data['radius'] *= 1000
    data['depth'] *= 1000
    data['density'] *= 1000
    return data.iloc[::-1]
    
if __name__ == '__main__':
    main()