import numpy as np
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
import pandas as pd

G = 6.6743e-11
M = 5.9722e24
R = 6.371e6
c = 3e8


def main():
    v = np.sqrt(G * M / R)
    print('Aw = v_max = ', v)
    beta = v/c
    print('beta = ', beta)
    gamma = 1 / np.sqrt(1 - beta**2)
    print('gamma = ', gamma)



if __name__ == '__main__':
    main()