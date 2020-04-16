# Automatica example with constant estimates based on sampling

import casadi as ca
import numpy as np
import scipy as sp
import control.matlab
import control as cn

import matplotlib.pyplot as plt
import matplotlib as mpl

import params as par
from params import *
from integrator import Integrator
from ocp_solver import OcpSolver, LqrSolver
from estimate_constants import *
import json

import pprint
import datetime

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
font = {'family':'serif'}
plt.rc('font',**font)

# LQR_INIT = True
LQR_INIT = False
# FLIP_LQR_INIT = True
FLIP_LQR_INIT = False
# UPDATE_FIGURES = True
UPDATE_FIGURES = False
# ESTIMATE_CONSTANTS = True
ESTIMATE_CONSTANTS = False
RUN = True
# RUN = False
SAVE_RESULTS_JSON = True
# SAVE_RESULTS_JSON = False 


nsim = int(par.Tf/par.Ts)
warmstart_iters = 0

x0 = par.x0_v[0]
n_scenarios = len(par.x0_v)

# estimate constants
if ESTIMATE_CONSTANTS:
    estimate_constants(update_json = True)

const = json.load(open('constants.json','r'))

# compute other constants

a1 = const['a1']
a2 = const['a2']
a3 = const['a3']
if a1 < 0 or a2 < 0 or a3 < 0:
    raise Exception('One of the a constants is \
        negative (a1 = {}, a2 = {}, a3 = {})'.format(a1, a2, a3))

a_bar = const['a3']/const['a2']
sigma = const['sigma']
mu_tilde = const['mu_tilde']
kappa_hat = const['kappa_hat']
L_phi_x = const['L_phi_x']
L_phi_u = const['L_phi_u']

L_psi_x = L_phi_x * np.exp(par.Ts * L_phi_x)
L_psi_u = L_phi_u * np.exp(par.Ts * L_phi_x)

eta   = L_psi_u + L_psi_x*sigma
theta = L_psi_u

kappa = kappa_hat*(1+par.Ts*sigma*theta)

pprint.pprint(const)
max_ts = (1 - kappa_hat)/(kappa_hat*sigma*theta)
if kappa > 1.0: 
    pprint.pprint(const)
    print('kappa = {}'.format(kappa))
    print('maximum Ts given current constants is {}'.format(max_ts))
    raise Exception('kappa > 1.0!')

gamma = sigma*kappa_hat*eta
gamma_hat = gamma/np.sqrt(a1)
mu_hat = L_phi_u * np.exp(par.Ts * L_phi_x) * mu_tilde

beta = 0.5*a_bar/(2*gamma_hat)
print('beta = {}'.format(beta))

lower_bound = par.Ts*mu_hat/(1-kappa)
upper_bound = (1-(1-par.Ts*a_bar)**(1.0/2.0))/(par.Ts*gamma_hat)
if  lower_bound > upper_bound:
    raise Exception('beta: lower bound greater than upper bound!')

# temp code
n_points = 100
tmin = -8
tmax = -2
T = np.logspace(tmin,tmax, n_points)
lower_bound = np.linspace(tmin, tmax, n_points)
upper_bound = np.linspace(tmin, tmax, n_points)
for i in range(n_points):
    lower_bound[i] = T[i]*mu_hat/(1-kappa)
    upper_bound[i] = (1-(1-T[i]*a_bar)**(1.0/2.0))/(T[i]*gamma_hat)

plt.figure()
plt.plot(T, lower_bound)
plt.plot(T, upper_bound)
plt.plot(par.Ts, w2, '*')
plt.plot([T[0], T[-1]], [beta, beta], '--')
plt.legend(['lower', 'upper', 'w2', 'beta'])
plt.show()

max_Ts_beta = beta*(1-kappa)/mu_hat
if par.Ts > max_Ts_beta: 
    raise Exception('Ts = {} bigger than max_Ts_beta = {}'.format(par.Ts, max_Ts_beta))

if beta < 0.0:
    pprint.pprint(const)
    raise Exception('beta is negative!')

# run simulation
fig1, (ax11) = plt.subplots(1,1, figsize=(3.0,3.0))
fig2, (ax21, ax22) = plt.subplots(2,1, figsize=(3.0,6.0))
fig3, (ax31, ax32, ax33) = plt.subplots(3,1, figsize=(3.0,6.0))

if RUN:
    ocp_solver = OcpSolver()
    lqr_solver = LqrSolver()
    ocp_solver.nlp_eval()

    XSIM = np.zeros((nsim+1, 2, n_scenarios))
    XSIMEXACT = np.zeros((nsim+1, 2, n_scenarios))
    USIM = np.zeros((nsim, 1, n_scenarios))

    VSIMSQRT = np.zeros((nsim, 1, n_scenarios))
    ZSIM = np.zeros((nsim, 1, n_scenarios))
    DELTAZSIM = np.zeros((nsim, 1, n_scenarios))
    VSOSIM = np.zeros((nsim, 1, n_scenarios))

    for j in range(len(par.x0_v)):
        x0 = par.x0_v[j]
        XSIM[0,:,j] = x0
        XSIMEXACT[0,:,j] = x0

        # closed loop simulation
        integrator = Integrator(par.ode)

        ocp_solver = OcpSolver()

        ocp_solver.rti_eval()
        ocp_solver.nlp_eval()

        if LQR_INIT:
            if FLIP_LQR_INIT: 
                x0_lqr = -par.x0_v[j]
            else:
                x0_lqr = par.x0_v[j]

            lqr_solver.update_x0(x0_lqr)
            lqr_solver.lqr_eval()
            lqr_sol = lqr_solver.get_lqr_sol()
            ocp_solver.set_sol_lin(lqr_sol)

        if warmstart_iters > 0:
            ocp_solver.update_x0(XSIM[0,:,j])
            for i in range(warmstart_iters):
                ocp_solver.rti_eval()

        ocp_solver.update_x0(XSIM[0,:,j])
        for i in range(nsim):

            # update state in OCP solver
            ocp_solver.update_x0(XSIM[i,:,j])

            # get primal dual solution (equalities only)
            rti_sol = ocp_solver.get_rti_sol()
            lam_g = rti_sol['lam_g'].full()
            x = rti_sol['x'].full()
            z = np.vstack((x, lam_g))
            ZSIM[i, 0] = np.linalg.norm(z)

            # get V value (this requires solving exactly the NLP)
            exact_sol = ocp_solver.nlp_eval()
            VSIMSQRT[i,0,j] = np.sqrt(exact_sol['f'].full())

            # get primal dual exact sol
            exact_lam_g = exact_sol['lam_g'].full()
            exact_x = exact_sol['x'].full()
            exact_u = exact_sol['x'].full()[2]
            exact_z = np.vstack((exact_x, exact_lam_g))
            DELTAZSIM[i,0,j] = np.linalg.norm(z - exact_z)
            VSOSIM[i,0,j] = VSIMSQRT[i,0,j] + beta*DELTAZSIM[i,0,j] 

            # call solver
            solver_out = ocp_solver.rti_eval()
            status = ocp_solver.get_status()
            if status != 'Solve_Succeeded':
                raise Exception('Solver returned status {} at iteration {}'\
                            .format(status, i))

            # get first control move
            u = solver_out['x'].full()[2]
            USIM[i,0,j] = u

            XSIM[i+1,:,j] = integrator.eval(par.Ts, XSIM[i,:,j], \
                u).full().transpose() 

            # get first control move (exact)
            ocp_solver.update_x0(XSIMEXACT[i,:,j])
            status = ocp_solver.get_status()
            exact_sol = ocp_solver.nlp_eval()
            if status != 'Solve_Succeeded':
                raise Exception('Solver returned status {} at iteration {}'\
                            .format(status, i))

            exact_u = exact_sol['x'].full()[2]

            XSIMEXACT[i+1,:,j] = integrator.eval(par.Ts, XSIMEXACT[i,:,j], \
                    exact_u).full().transpose() 
            print(u, exact_u)
            print(XSIM[i,:], XSIMEXACT[i,:])

else:

    res = json.load(open('results/results.json','r'))
    const = res['const']
    DELTAZSIM = np.array(res['DELTAZSIM'])
    VSOSIM = np.array(res['VSOSIM'])
    XSIM = np.array(res['XSIM'])
    XSIMEXACT = np.array(res['XSIMEXACT'])
    VSIMSQRT = np.array(res['VSIMSQRT'])

# plot
time = np.linspace(0, par.Ts*nsim, nsim)

DELTADELTAZSIM = np.zeros((nsim-1, 1))
DELTAVSIMSQRT = np.zeros((nsim-1, 1))
DELTAVSOSIM = np.zeros((nsim-1, 1))

ax11.grid()
ax21.grid()
ax22.grid()
ax31.grid()
ax32.grid()
ax33.grid()

for j in range(n_scenarios):
    ax11.plot(XSIM[:,1,j], XSIM[:,0,j])
    ax11.set_xlabel(r"$x_1$")
    ax11.set_ylabel(r"$x_2$")
    ax11.set_xlim([-1.0, 1.0])
    ax11.set_ylim([-1.0, 1.0])

    ax21.semilogy(VSIMSQRT[:,0,j], DELTAZSIM[:,0,j])
    ax21.set_xlabel(r"$V(x)^{\frac{1}{q}}$")
    ax21.set_ylabel(r"$\| z - \bar{z}(x)\|$")

    ax22.plot(time, VSOSIM[:,0,j])
    ax22.set_xlabel(r"time [s]")
    ax22.set_ylabel(r"$V_{\mathrm{so}}(x,z)$")


    for i in range(nsim-1):
        DELTADELTAZSIM[i,0] = DELTAZSIM[i,0,j] - DELTAZSIM[i+1, 0, j]
        DELTAVSIMSQRT[i,0] = VSIMSQRT[i,0,j] - VSIMSQRT[i+1, 0, j]
        DELTAVSOSIM[i,0] = VSOSIM[i,0,j] - VSOSIM[i+1, 0, j]

    ax31.semilogy(time[0:-1], DELTADELTAZSIM)
    ax31.set_xlabel(r"time [s]")
    ax31.set_ylabel(r"$\delta \| z - \bar{z}(x) \|$")

    ax32.semilogy(time[0:-1], DELTAVSIMSQRT)
    ax32.set_xlabel(r"time [s]")
    ax32.set_ylabel(r"$\delta \sqrt{V(x)}$")

    ax33.semilogy(time[0:-1], DELTAVSOSIM)
    ax33.set_xlabel(r"time [s]")
    ax33.set_ylabel(r"$\delta V_{\mathrm{so}}(x,z)$")

if UPDATE_FIGURES:
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig('../../notes/Figures/chen_states.pdf', dpi=300, \
        bbox_inches="tight")
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig('../../notes/Figures/chen_lyapunov.pdf', dpi=300, bbox_inches="tight")

def np_array_to_list(np_array):
    if isinstance(np_array, (np.ndarray)):
        return np_array.tolist()
    else:
        raise(Exception(
            "Cannot convert to list type {}".format(type(np_array))))
plt.show()

if SAVE_RESULTS_JSON and RUN:
    data_and_time = str(datetime.datetime.now())
    res = dict()
    res['const'] = const
    res['DELTAZSIM'] = DELTAZSIM
    res['VSOSIM'] = VSOSIM
    res['XSIM'] = XSIM
    res['XSIMEXACT'] = XSIMEXACT
    res['VSIMSQRT'] = VSIMSQRT


    with open('results/results' + data_and_time + '.json', 'w') as outfile:
        json.dump(res, outfile, default=np_array_to_list)
