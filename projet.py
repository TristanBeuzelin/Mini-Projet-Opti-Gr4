from casadi import *
import time

# Initialisation variables
m = 5
p = 3
alpha = 0.1
c = MX([30e-3, 1e-3, 4e-3, 1e-3, 1e-3])  # Erreur sujet (manque dernier coeff c)
v = MX([0.9, 1.5, 1.1])
d = MX([400, 67, 33])
A = MX(np.array([[3.5, 2., 1.], [250., 80., 25.], [0., 8., 3.], [0., 40., 10.], [0., 8.5, 0.]]))

# Création problème d'optimisation
opti = casadi.Opti()
r = opti.variable(m)
q = opti.variable(p)

h = ( ( q*exp(-alpha*q) ) + ( d*exp(-alpha*d) ) ) / ( exp(-alpha*q) + exp(-alpha*d) )
f = -( dot(v,h) - dot(c,r) )

opti.minimize(f)
opti.subject_to(A@q-r<=0)
opti.subject_to(-q<=0)
opti.subject_to(-r<=0)
r0 = np.zeros((m, 1))
opti.set_initial(r,r0)
q0 = np.zeros((p, 1))
opti.set_initial(q, q0)
opti.solver('ipopt')
sol = opti.solve()
print("q optimal : ", sol.value(q))
print("r optimal : ", sol.value(r))