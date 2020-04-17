import casadi as ca
import numpy as np
import scipy as sp
import control.matlab
import control as cn

import matplotlib.pyplot as plt
import matplotlib as mpl

import params as par
from integrator import Integrator
from ocp_solver import OcpSolver, LqrSolver
import json

LQR_INIT = True

def estimate_constants(update_json=False):
    # estimate L_phi_x, L_phi_u using max norm of Jacobians
    # max attained at x1 = 1, x2 = -1
    L_phi_u = np.sqrt((par.mu + (1 - par.mu))**2 + \
            (par.mu -4*(1-par.mu))**2)
    
    n_points = 100
    u_grid = np.linspace(-par.umax, +par.umax, n_points)
    norm_dphi_dx = np.zeros((n_points,1))
    for i in range(n_points):
        u = u_grid[i]
        dphi_dx = np.array([[u*(1-par.mu), 1],[1, -4*u*(1-par.mu)]])
        norm_dphi_dx[i] = np.linalg.norm(dphi_dx)

    L_phi_x = np.max(norm_dphi_dx)

    nsim = int(par.Tf/par.Ts)
    x0 = par.x0_v[0]
    n_scenarios = len(par.x0_v)
    n_sqp_it = 3

    # estimate a_1, a_2, a_3, sigma, mu_tilde constants based on sampling
    VEXACT       = np.zeros((nsim,1,n_scenarios))
    SOLEXACTNORM = np.zeros((nsim,1,n_scenarios))
    DELTAVEXACT  = np.zeros((nsim-1,1,n_scenarios))
    XNORMSQ      = np.zeros((nsim,1,n_scenarios))
    XNORM        = np.zeros((nsim,1,n_scenarios))
    XSIMEXACT    = np.zeros((nsim+1,2,n_scenarios))
    USIMEXACT    = np.zeros((nsim,1,n_scenarios))

    exact_ocp_solver = OcpSolver()

    for j in range(n_scenarios):
        x0 = par.x0_v[j]

        # closed loop simulation
        XSIMEXACT[0,:,j] = x0

        integrator = Integrator(par.ode)

        exact_ocp_solver = OcpSolver()

        exact_ocp_solver.nlp_eval()

        for i in range(nsim):

            XNORMSQ[i,0,j] = np.linalg.norm(XSIMEXACT[i,:,j])**2
            XNORM[i,0,j] = np.linalg.norm(XSIMEXACT[i,:,j])

            # update state in OCP solver
            exact_ocp_solver.update_x0(XSIMEXACT[i,:,j])

            solver_out = exact_ocp_solver.nlp_eval()
            status = exact_ocp_solver.get_status()
            if status != 'Solve_Succeeded':
                raise Exception('Solver returned status {} at iteration {}'\
                        .format(status, i))

            # get primal-dual solution
            x = solver_out['x'].full()
            lam_g = solver_out['lam_g'].full()
            SOLEXACTNORM[i,:,j] = np.linalg.norm(np.vstack([x, lam_g])) 
            # get first control move
            u = solver_out['x'].full()[2]
            USIMEXACT[i,0,j] = u

            VEXACT[i,0,j] = solver_out['f'].full()

            XSIMEXACT[i+1,:,j] = integrator.eval(par.Ts, XSIMEXACT[i,:,j], \
                u).full().transpose() 

        for i in range(nsim-1):
            DELTAVEXACT[i,0,j] = VEXACT[i+1,0,j] - VEXACT[i,0,j]

    XSIMEXACT = XSIMEXACT[0:-1,:,:]

    VEXACT_OVER_XNORMSQ = []
    DELTAVEXACT_OVER_XNORMSQ = []
    SOLEXACTNORM_OVER_XNORM = []
    VEXACTSQRT_OVER_XNORM = []
    for j in range(n_scenarios):
        VEXACT_OVER_XNORMSQ.append(np.divide(VEXACT[:,0,j], XNORMSQ[:,0,j]))
        DELTAVEXACT_OVER_XNORMSQ.append(np.divide(DELTAVEXACT[:,0,j], \
            XNORMSQ[0:-1,0,j]))
        SOLEXACTNORM_OVER_XNORM.append(np.divide( \
            SOLEXACTNORM[:,0,j], XNORM[:,0,j]))
        VEXACTSQRT_OVER_XNORM.append(np.divide(np.sqrt(VEXACT[:,0,j]), \
            XNORM[:,0,j])) 

    # compute constants
    a1   = np.min(VEXACT_OVER_XNORMSQ) 
    a2   = np.max(VEXACT_OVER_XNORMSQ) 
    a3   = np.min(-1.0/par.Ts*np.array(DELTAVEXACT_OVER_XNORMSQ))

    sigma = np.max(SOLEXACTNORM_OVER_XNORM)
    mu_tilde = np.max(VEXACTSQRT_OVER_XNORM)

    print('a1 = {}, a2 = {}, a3 = {}, sigma = {}, mu_tilde = {}' \
        .format(a1, a2, a3, sigma, mu_tilde))

    if a1 < 0 or a2 < 0 or a3 < 0:
        raise Exception('One of the a constants is \
            negative (a1 = {}, a2 = {}, a3 = {})'.format(a1, a2, a3))

    # estimate \hat{\kappa}

    ocp_solver = OcpSolver()
    lqr_solver = LqrSolver()
    ocp_solver.nlp_eval()

    XSIM = np.zeros((nsim+1, 2, n_scenarios))
    USIM = np.zeros((nsim, 1, n_scenarios))

    VSIMSQRT = np.zeros((nsim, 1, n_scenarios))
    ZSIM = np.zeros((nsim, 1, n_scenarios))
    DELTAZSIM = np.zeros((nsim, 1, n_scenarios))
    VSOSIM = np.zeros((nsim, 1, n_scenarios))

    kappa = 0.0

    for j in range(len(par.x0_v)):
        x0 = par.x0_v[j]
        XSIM[0,:,j] = x0

        # closed loop simulation
        integrator = Integrator(par.ode)

        ocp_solver = OcpSolver()

        ocp_solver.rti_eval()
        ocp_solver.nlp_eval()


        ocp_solver.update_x0(XSIM[0,:,j])
        for i in range(nsim):

            # update state in OCP solver
            ocp_solver.update_x0(XSIM[i,:,j])
            DELTAZSIM = np.zeros((n_sqp_it, 1))

            if LQR_INIT:
                lqr_solver.update_x0(XSIM[i,:,j])
                lqr_solver.lqr_eval()
                lqr_sol = lqr_solver.get_lqr_sol()
                ocp_solver.set_sol_lin(lqr_sol)

            for k in range(n_sqp_it):

                # get primal dual solution (equalities only)
                rti_sol = ocp_solver.get_rti_sol()
                lam_g = rti_sol['lam_g'].full()
                x = rti_sol['x'].full()
                z = np.vstack((x, lam_g))

                # get primal dual exact sol
                exact_sol = ocp_solver.nlp_eval()
                nlp_sol = ocp_solver.get_nlp_sol()
                exact_lam_g = nlp_sol['lam_g'].full()
                exact_x = nlp_sol['x'].full()
                exact_z = np.vstack((exact_x, exact_lam_g))
                DELTAZSIM[k,0] = np.linalg.norm(z - exact_z)

                # call solver
                solver_out = ocp_solver.rti_eval()
                status = ocp_solver.get_status()
                if status != 'Solve_Succeeded':
                    raise Exception('Solver returned status {} at iteration {}'\
                            .format(status, i))

            # estimate \hat{kappa}
            KAPPAESTIMATE = np.zeros((n_sqp_it,1))
            for k in range(n_sqp_it-1):
                KAPPAESTIMATE[k,0] = DELTAZSIM[k+1,0]/(DELTAZSIM[k,0])

            kappa = np.max([np.max(KAPPAESTIMATE), kappa])
             
            # get first control move
            u = solver_out['x'].full()[2]
            USIM[i,0,j] = u

            XSIM[i+1,:,j] = integrator.eval(par.Ts, XSIM[i,:,j], \
                u).full().transpose() 

    print('kappa = {}'.format(kappa))

    if kappa > 1.0:
        raise Exception('kappa bigger than 1.0!')

    const = dict()
    const['a1'] = a1
    const['a2'] = a2
    const['a3'] = a3
    const['sigma'] = sigma
    const['kappa_hat'] = kappa
    const['mu_tilde'] = mu_tilde
    const['L_phi_x'] = L_phi_x
    const['L_phi_u'] = L_phi_u

    if update_json:
        with open('constants.json', 'w') as outfile:
            json.dump(const, outfile)

    return const
