from casadi import *
import time

opti = casadi.Opti();
m = 5
p = 3
alpha = 0.1
c = SX([30e-3, 1e-3, 4e-3, 1e-3])
v = SX([0.9, 1.5, 1.1])
d = SX([400, 67, 33])
A = SX([[3.5, 2, 1], [250, 80, 25], [0, 8, 3], [0, 40, 10], [0, 8.5, 0]])
r = opti.variable(m)
q = opti.variable(p)
print(q@exp(-alpha*q).T)
print(d@exp(-alpha*d).T)
h = ((q@exp(-alpha*q).T)+(d@exp(-alpha*d).T))/(exp(-alpha*q)+exp(-alpha*d))
f = c.T@r - v.T@h
opti.minimize(f)
opti.subject_to(A@q-r<=0)
opti.subject_to(-q<=0)
opti.subject_to(-r<=0)
r0 = np.zeros((m, 1))
opti.set_initial(r,r0)
q0 = np.zeros((p, 1))
opti.set_initial(q, q0)
opti.solver('ipopt');
sol = opti.solve();
print(sol.value(r, q))
