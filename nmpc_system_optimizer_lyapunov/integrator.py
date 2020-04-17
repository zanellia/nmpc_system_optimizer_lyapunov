import casadi as ca
import params 

class Integrator():

    def __init__(self, xdot):

        self.x = ca.MX.sym('x', 2, 1)
        self.u = ca.MX.sym('u', 1, 1)
        self.Ts = ca.MX.sym('Ts', 1, 1)
        self.xdot = xdot

        def discretize_dynamics(xdot, Ts, x, u):

            M  = 1 # RK4 steps per interval
            DT = Ts/M
            f  = ca.Function('f', [x, u], [xdot])
            X0 = ca.MX.sym('X0', 2, 1)
            U  = ca.MX.sym('U', 1, 1)
            X  = X0

            for j in range(M):
                k1 = f(X, U)
                k2 = f(X + DT/2 * k1, U)
                k3 = f(X + DT/2 * k2, U)
                k4 = f(X + DT * k3, U)
                X=X+DT/6*(k1 + 2*k2 + 2*k3 +k4)

            xplus = ca.Function('x_plus', [Ts, X0, U], [X])

            return xplus

        self.__discrete_dynamics = \
            discretize_dynamics(xdot(self.x, self.u), \
            self.Ts, self.x, self.u)

    def eval(self, Ts_, x_, u_):

        if Ts_ <= 0:
            raise Exception('sampling time cannot be negative nor zero!')
        xplus = self.__discrete_dynamics(Ts_, x_, u_) 
        return xplus

