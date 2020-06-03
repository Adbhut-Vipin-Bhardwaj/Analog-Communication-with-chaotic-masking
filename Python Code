import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma,r,b = (16, 45.6, 4);

#power spectrum of msg should be significantly less than y[0] (which is used
#for masking)
def msg(t):
    return 1e-3*np.sin(100*t);

y_init = np.array([1,1,1,1.1,1.1,1.1]);
def derivs(t,y):
    return (sigma*(y[1]-y[0]), r*y[0]-y[1]-20*y[0]*y[2], 5*y[0]*y[1]-b*y[2], 
            sigma*(y[4]-y[3]), r*(y[0]+msg(t))-y[4]-20*(y[0]+msg(t))*y[5], 
            5*(y[0]+msg(t))*y[4]-b*y[5]);

soln = solve_ivp(derivs, t_span=(0.0, 20.0), y0=y_init, dense_output=True);

time_points = np.linspace(start=0, stop=20, num=100);
error = soln.sol(time_points)[0] - soln.sol(time_points)[3];
plt.plot(time_points, error);
