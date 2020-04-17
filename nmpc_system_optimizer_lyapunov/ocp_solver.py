import params as par
import numpy as np
import casadi as ca
import control as cn
from integrator import Integrator

class OcpSolver():

    def __init__(self):

        Td = par.Tp/par.N

        # linearize model
        X_lin = ca.MX.sym('X_lin', 2, 1)
        U_lin = ca.MX.sym('U_lin', 1, 1)
        A_c = ca.Function('A_c', \
            [X_lin, U_lin], [ca.jacobian(par.ode(X_lin, U_lin), X_lin)])

        A_c = A_c([0, 0], 0).full()
        B_c = ca.Function('B_c', \
            [X_lin, U_lin], [ca.jacobian(par.ode(X_lin, U_lin), U_lin)])

        B_c = B_c([0, 0], 0).full()

        # solve continuous-time Riccati equation
        Qt, e, K = cn.care(A_c, B_c, par.Q, par.R)

        # this is the value used in Chen1998!! 
        # they do not use LQR, but a modified linear controller
        # Qt = np.array([[16.5926, 11.5926], [11.5926, 16.5926]])

        self.integrator = Integrator(par.ode)
        self.x = self.integrator.x
        self.u = self.integrator.u
        self.Td = Td
        
        # start with an empty NLP
        w=[]
        w0 = []
        lbw = []
        ubw = []
        g=[]
        lbg = []
        ubg = []
        Xk = ca.MX.sym('X0', 2, 1)
        w += [Xk]
        lbw += [0, 0]
        ubw += [0, 0]
        w0 += [0, 0]
        f = 0

        # formulate the NLP
        for k in range(par.N):

            # new NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), 1, 1)

            f = f + Td*ca.mtimes(ca.mtimes(Xk.T, par.Q), Xk) \
                    + Td*ca.mtimes(ca.mtimes(Uk.T, par.R), Uk)

            w   += [Uk]
            lbw += [-par.umax]
            ubw += [par.umax]
            w0  += [0.0]

            # integrate till the end of the interval
            Xk_end = self.integrator.eval(Td, Xk, Uk)

            # new NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), 2, 1)
            w   += [Xk]
            lbw += [-np.inf, -np.inf]
            ubw += [  np.inf,  np.inf]
            w0  += [0, 0]

            # add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0, 0]
            ubg += [0, 0]

        f = f + ca.mtimes(ca.mtimes(Xk_end.T, Qt), Xk_end)
        g = ca.vertcat(*g)
        w = ca.vertcat(*w)

        # create an NLP solver
        prob = {'f': f, 'x': w, 'g': g}
        self.__nlp_solver = ca.nlpsol('solver', 'ipopt', prob);
        self.__lbw = lbw
        self.__ubw = ubw
        self.__lbg = lbg
        self.__ubg = ubg
        self.__w0 = np.array(w0)
        self.__sol_lin = np.array(w0).transpose()
        self.__rti_sol = []
        self.__nlp_sol = []

        # create QP solver
        nw = len(w0)
        # define linearization point
        w_lin = ca.MX.sym('w_lin', nw, 1)
        w_qp = ca.MX.sym('w_qp', nw, 1)

        # linearized objective = original LLS objective
        f_lin = ca.substitute(f, w, w_qp) + par.alpha*ca.dot(w_qp - w_lin, w_qp - w_lin)

        nabla_g = ca.jacobian(g, w).T
        g_lin = ca.substitute(g, w, w_lin) + \
            ca.mtimes(ca.substitute(nabla_g, w, w_lin).T, w_qp - w_lin) 

        # create a QP solver
        prob = {'f': f_lin, 'x': w_qp, 'g': g_lin, 'p' : w_lin}
        self.__rti_solver = ca.nlpsol('solver', 'ipopt', prob);

    def update_x0(self, x0):
        self.__lbw[0] = x0[0]
        self.__lbw[1] = x0[1]
        self.__ubw[0] = x0[0]
        self.__ubw[1] = x0[1]

    def nlp_eval(self):

        sol = self.__nlp_solver(x0=self.__w0, lbx=self.__lbw, ubx=self.__ubw,\
                lbg=self.__lbg, ubg=self.__ubg)

        self.__status = self.__nlp_solver.stats()['return_status']

        self.__nlp_sol = sol

        return sol

    def rti_eval(self):

        # solve QP obtained linearizing the NLP at the 
        # current linearization point self.__rti_sol

        sol = self.__rti_solver(x0=self.__w0, lbx=self.__lbw, \
            ubx=self.__ubw, lbg=self.__lbg, ubg=self.__ubg,
            p = self.__sol_lin)

        # update current sol (i.e. linearization point)
        self.__sol_lin = sol['x'].full().transpose()

        self.__status = self.__rti_solver.stats()['return_status']

        self.__rti_sol = sol

        return sol

    def get_status(self):
        return self.__status

    def get_rti_sol(self):
        return self.__rti_sol

    def get_nlp_sol(self):
        return self.__nlp_sol

    def set_sol_lin(self, sol_lin):
        self.__sol_lin = sol_lin['x'].full()
        self.__rti_sol = sol_lin

class LqrSolver():

    def __init__(self, x_lin=[0,0], u_lin=0):

        Td = par.Tp/par.N

        # linearize model
        X_lin = ca.MX.sym('X_lin', 2, 1)
        U_lin = ca.MX.sym('U_lin', 1, 1)
        A_c = ca.Function('A_c', \
            [X_lin, U_lin], [ca.jacobian(par.ode(X_lin, U_lin), X_lin)])

        A_c = A_c([0, 0], 0).full()
        B_c = ca.Function('B_c', \
            [X_lin, U_lin], [ca.jacobian(par.ode(X_lin, U_lin), U_lin)])

        B_c = B_c([0, 0], 0).full()

        # solve continuous-time Riccati equation
        Qt, e, K = cn.care(A_c, B_c, par.Q, par.R)

        self.integrator = Integrator(par.ode)
        self.x = self.integrator.x
        self.u = self.integrator.u
        self.Td = Td
        
        # start with an empty NLP
        w=[]
        w0 = []
        w_lin = []
        lbw = []
        ubw = []
        g=[]
        lbg = []
        ubg = []
        Xk = ca.MX.sym('X0', 2, 1)
        w += [Xk]
        lbw += [0, 0]
        ubw += [0, 0]
        w0 += [0, 0]
        w_lin += x_lin
        f = 0

        # formulate the NLP
        for k in range(par.N):

            # new NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), 1, 1)

            f = f + Td*ca.mtimes(ca.mtimes(Xk.T, par.Q), Xk) \
                    + Td*ca.mtimes(ca.mtimes(Uk.T, par.R), Uk)

            w      += [Uk]
            lbw    += [-np.inf]
            ubw    += [np.inf]
            w0     += [0.0]
            w_lin  += [u_lin]

            # integrate till the end of the interval
            Xk_end = self.integrator.eval(Td, Xk, Uk)

            # new NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), 2, 1)
            w      += [Xk]
            lbw    += [-np.inf, -np.inf]
            ubw    += [  np.inf,  np.inf]
            w0     += [0, 0]
            w_lin  += x_lin

            # add equality constraint
            g   += [Xk_end-Xk]
            lbg += [0, 0]
            ubg += [0, 0]

        f = f + ca.mtimes(ca.mtimes(Xk_end.T, Qt), Xk_end)
        g = ca.vertcat(*g)
        w = ca.vertcat(*w)

        # create an NLP solver
        self.__lbw = lbw
        self.__ubw = ubw
        self.__lbg = lbg
        self.__ubg = ubg
        self.__w0 = np.array(w0)
        self.__lqr_sol = []

        # create QP solver
        nw = len(w0)
        # define linearization point
        # w_lin = ca.MX.sym('w_lin', nw, 1)
        w_qp = ca.MX.sym('w_qp', nw, 1)

        # linearized objective = original LLS objective
        f_lin = ca.substitute(f, w, w_qp)

        nabla_g = ca.jacobian(g, w).T
        g_lin = ca.substitute(g, w, w_lin) + \
            ca.mtimes(ca.substitute(nabla_g, w, w_lin).T, w_qp - w_lin) 

        # create a QP solver
        prob = {'f': f_lin, 'x': w_qp, 'g': g_lin}
        self.__lqr_solver = ca.nlpsol('solver', 'ipopt', prob);

    def update_x0(self, x0):
        self.__lbw[0] = x0[0]
        self.__lbw[1] = x0[1]
        self.__ubw[0] = x0[0]
        self.__ubw[1] = x0[1]

    def lqr_eval(self):

        sol = self.__lqr_solver(x0=self.__w0, lbx=self.__lbw, ubx=self.__ubw,\
                lbg=self.__lbg, ubg=self.__ubg)

        self.__status = self.__lqr_solver.stats()['return_status']

        self.__lqr_sol = sol

        return sol

    def get_status(self):
        return self.__status

    def get_lqr_sol(self):
        return self.__lqr_sol


