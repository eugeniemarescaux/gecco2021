import sys
sys.path.append('/home/eugenie/Documents/code-mo/python/')
import pymoo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.integrate as integrate
from pymoo.factory import get_performance_indicator
import os
import cma
import time
import json
import argparse
import random

# Definition of the different kinds of Pareto fronts
def f(x,name,r2):
    if x < 0 or x > 1:
        return r2
    # Convex functions
    if name == 'convex-biL':
        e = np.exp(1)
        return e/(e-1) * np.exp(-x) + 1 - e/(e-1)
    elif name == 'convex-doublesphere': 
        return 1 + x - 2*np.sqrt(x)
    elif name == 'convex-zdt1':
        return 1 - np.sqrt(x)
    # Concave functions
    elif name == 'concave-biL': # bi-Lipschitz
        return 1 - x/2 - x**2/2
    elif name == 'concave-dtlz2':
        if 1 - x**2 >= 0:
            return np.sqrt(1-x**2)
        else:
            return r2
    elif name == 'concave-zdt2':
        return 1 - x**2
    # Linear functions
    elif name == 'linear':
        return 1 - x


# Tool for computing the p-optimal distribution
def dist_from_diff(delta, name):
    if name == 'concave-dtlz2':
        return dist_from_diff_dtlz2(delta)
    else:
        return dist_from_diff_normal(delta)
def dist_from_diff_normal(delta):
    xopt = []
    xopti=0
    for d in delta[:-1]:
        xopti += d
        xopt.append(xopti)
    xopt.append(1-delta[-1])
    return xopt
def dist_from_diff_dtlz2(delta):    
    xopt = []
    xopti=0
    for d in delta[:-1]:
        xopti += d
        xopt.append(xopti)
    xopt.append(1-np.sign(delta[-1])*(np.abs(delta[-1]))**2)
    return xopt
def penalization(delta, name):
    pop = dist_from_diff(delta,name)
    pen = 0
    for x in pop:
        if x > 1:
            pen += (x-1)**2
        if x < 0:
            pen += x**2
    return pen

def main():
    """Main function, to get the arguments and computing the p-optimal distributions."""
    # Get parameters
    parser = argparse.ArgumentParser(description='Obtain the parameters.\n')
    parser.add_argument('--pmin', type=int, default=1, help="The p for which Sp is computed should be >= pmin")
    parser.add_argument('--pmax', type=int, default=100,help="The p for which Sp is computed should be <= pmax")
    parser.add_argument('--fun', type=str, default='convex-biL', help="The function f describing the Pareto front." 
                        +"Possibilities are : 'convex-biL', 'convex-doublesphere', 'convex-zdt1', 'concave-biL',"
                        +"'concave-dtlz2', 'concave-zdt2' and 'linear'")
    parser.add_argument('--nb_restarts', type=int, default=1,help="Number of runs to compute Sp for p fixed.")
    parser.add_argument('--r1', type=float, default=1,help="First coordinate of the reference point.")
    parser.add_argument('--r2', type=float, default=1,help="Second coordinate of the reference point.")
    parser.add_argument('--opt', type=str, default='CMA',help="Choice of the"
    +"optimizer to compute the p-optimal distribution. Either 'CMA' or 'SLSQP'.")
    args = parser.parse_args()

    # Create the required directories
    new_dir = "Data-popt/"
    path = os.getcwd() + "/Data-popt/"
    if not os.path.exists(path):
        os.mkdir(path)

    # Compute the list of p-values for which to compute Sp, the p-optimal
    # distribution
    imin = int(np.floor(np.log(args.pmin)/np.log(10)))
    imax = int(np.ceil(np.log(args.pmax)/np.log(10)))
    allp = [int(10**(i+j/10)) for i in range(imin, imax+1) for j in range(10)
            if int(10**(i+j/10)) >= args.pmin and int(10**(i+j/10)) <= args.pmax] 
        

    # Define the hypervolume indicator w.r.t r
    hv = get_performance_indicator("hv", ref_point=np.array([args.r1,args.r2]))
    # Compute the hypervolume of the Pareto front
    hvPF = integrate.quad(lambda x:args.r2-f(x,args.fun,args.r2),0,1)[0] + (args.r1-1)*(args.r2-f(1,args.fun,args.r2))
    # Define the optimizer
    def obj_fun(delta):
        pop = dist_from_diff(delta,args.fun)
        return -hv.calc(np.array([[y,f(y,args.fun,args.r2)] for y in pop])) + penalization(delta,args.fun)
    def popt_CMA(p):
        diff0 = [1/(p+1) for i in range(p)]
        diffopt, es = cma.fmin2(obj_fun, diff0, 1e-3/p, {'bounds':[0,1], 'verbose':-9, 'verb_log':0, 'verb_disp':0})
        return dist_from_diff(diffopt,args.fun)
    def popt_SLSQP(p):
        diff0 = [1/(p+1) + (random.random() -0.5)/p/(p+1) for i in range(p)]
        res = minimize(obj_fun, diff0, method='SLSQP', bounds=[[0,1] for i in range(p)],options={'disp': False, 'ftol':1e-13,'maxiter':1000})
        return dist_from_diff(res.x,args.fun)
    def feasible(popt):
        for xopt in popt:
            if xopt < 0 or xopt > 1:
                return False
        return True

    # Compute the p-optimal distribution and store it
    for p in allp:
        print("p=",p)
        for restart in range(args.nb_restarts):
            t_begin = time.time()
            if p == 1:
                popt = popt_SLSQP(p)
            else:
                if args.opt == 'CMA':
                    popt = popt_CMA(p)
                elif args.opt == 'SLSQP':
                    popt = popt_SLSQP(p)
            t_computation = round(time.time() - t_begin,2)
            hvpopt = hv.calc(np.array([[y,f(y,args.fun,args.r2)] for y in popt]))
            data = {'fun': args.fun, 'p': p, 'popt': popt, 'hv': hvpopt,
            'gap':hvPF-hvpopt, 'time':t_computation, 'solver':args.opt,
            'r':(args.r1,args.r2), 'feas':feasible(popt)}
            end = '.txt'
            if args.r1 != 1 or args.r2 !=1:
                end = '_r1'+str(int(args.r1))+'_r2'+str(int(args.r2)) + end
            with open(new_dir + args.fun + end, 'a') as file:
                file.write(json.dumps(data) + '\n')

if __name__ == '__main__':
    main()
