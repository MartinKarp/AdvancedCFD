# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# %% Defining variables
mpl.style.use('ggplot')
t = 20
dtdx = 1/4
u = 1
#FOr the first task
def init_q1(t):
    q = np.ones((10,t))
    q[3:7,:] = 2
    return q
#For the second task
def init_q2(t):
    q = -1 * np.ones((10,t))
    q[3:7,:] = 2
    return q


# %% LAX method advection
q = init_q1(t)
for i in range(t-1):
    F = q[:,i]
    q[1:-1,i+1] = 0.5 * (q[:-2,i] + q[2:,i]) - 0.5 * dtdx * u * (F[2:] - F[:-2])
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Lax Advection')
plt.savefig('HW1/LAXadvection.eps', format = 'eps')

# %% LAX Burgers eq
q = init_q1(t)
print(q[:,0])
for i in range(t-1):
    F = 0.5*q[:,i]**2
    q[1:-1,i+1] = 0.5 * (q[:-2,i] + q[2:,i]) - 0.5 * dtdx * (F[2:] - F[:-2])
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Lax Burgers')
plt.savefig('HW1/LAXburger.eps', format = 'eps')

# %% LAX Wendroff advection
q = init_q1(t)
J = np.ones(9)
for i in range(t-1):
    F = q[:,i]
    q[1:-1,i+1] = q[1:-1,i] - 0.5 * dtdx * (F[2:] - F[:-2]) + 0.5 * dtdx**2 * (J[1:] * (F[2:] -F[1:-1]) - J[:-1] * (F[1:-1] -F[:-2]))
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Lax-Wendroff Advection')
plt.savefig('HW1/Wendroffadvection.eps', format = 'eps')


# %% LAX Wendroff Burgers
q = init_q1(t)
for i in range(t-1):
    J = np.ones(9) * 0.5 * (q[1:,i] + q[:-1,i])
    F = 0.5 * q[:,i]**2
    q[1:-1,i+1] = q[1:-1,i] - 0.5 * dtdx * (F[2:] - F[:-2]) + 0.5 * dtdx**2 * (J[1:] * (F[2:] -F[1:-1]) - J[:-1] * (F[1:-1] -F[:-2]))
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Lax-Wendroff Burgers')
plt.savefig('HW1/Wendroffburger.eps', format = 'eps')

# %% MacCormak advection
q = init_q1(t)
for i in range(t-1):
    F = q[:,i]
    qs = np.copy(q[:,i])
    qs[1:-1] -= dtdx * (F[2:] - F[1:-1])
    Fs = qs
    q[1:-1,i+1] = 0.5 * (q[1:-1,i] + qs[1:-1]) - 0.5 * dtdx * (Fs[1:-1] - Fs[:-2])
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('MacCormack Advection')
plt.savefig('HW1/Maccormackadvection.eps', format = 'eps')

# %% MacCormak burgers
q = init_q1(t)
for i in range(t-1):
    F = 0.5 * (q[:,i]**2)
    qs = np.copy(q[:,i])
    qs[1:-1] -= dtdx * (F[2:] - F[1:-1])
    Fs = 0.5 * (qs ** 2)
    q[1:-1,i+1] = 0.5 * (q[1:-1,i] + qs[1:-1]) - 0.5 * dtdx * (Fs[1:-1] - Fs[:-2])
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('MacCormack Burgers')
plt.savefig('HW1/Maccormackburgers.eps', format = 'eps')

# %% Godunov Advection
def godunov_adv(q,t):
    for i in range(t-1):
        q[1:-1,i+1] = q[1:-1,i] - dtdx * ( q[1:-1,i] - q[:-2,i])
    return q

q = init_q1(t)
q = godunov_adv(q,t)

plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Godunov Advection')
plt.savefig('HW1/Godunovadvection.eps', format = 'eps')

# %% Godunov Burgers
def godunov_burg(q,t):
    for i in range(t-1):
        f = np.zeros(9)
        F = 0.5 * q[:,i] ** 2
        for j in range(9):
            qj = q[j,i]
            qj1 = q[j+1,i]
            c = 0.5 * (qj +  qj1)
            if qj >  qj1:
                if c > 0:
                    f[j] = F[j]
                else:
                    f[j] = F[j+1]
            else:
                if( qj1 > 0 and qj < 0):
                    f[j] = 0
                elif(c > 0 and qj1 > qj and qj > 0):
                    f[j] = F[j]
                else:
                    f[j] = F[j+1]
        q[1:-1,i+1] = q[1:-1,i] - dtdx * ( f[1:] - f[:-1])
    return q

q = init_q1(t)
q = godunov_burg(q,t)
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Godunov Burgers')
plt.savefig('HW1/Godunovburgers.eps', format = 'eps')
# %% Roe Advection
def roe_adv(q,t):
    for i in range(t-1):
        f = np.zeros(9)
        F = q[:, i]
        A = 1
        f = 0.5 * (F[:-1] + F[1:]) - 0.5 * np.abs(A) * (q[1:,i] - q[:-1,i])
        q[1:-1,i+1] = q[1:-1,i] - dtdx * ( f[1:] - f[:-1])
    return q

q = init_q1(t)
q = roe_adv(q,t)
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('Roe Advection')
plt.savefig('HW1/Roeadvection.eps', format = 'eps')

# %% Roe Burgers
def roe_burg(q,t):
    for i in range(t-1):
        f = np.zeros(9)
        F = 0.5 * q[:, i] ** 2
        A = 0.5 * (q[ 1:, i] + q[:-1, i])
        f = 0.5 * (F[:-1] + F[1:]) - 0.5 * np.abs(A) * (q[1:,i] - q[:-1,i])
        q[1:-1,i+1] = q[1:-1,i] - dtdx * ( f[1:] - f[:-1])
    return q

q = init_q1(t)
q = roe_burg(q,t)
plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('Roe Burgers')
plt.savefig('HW1/Roeburgers.eps', format = 'eps')

# %% Roe Fromm Burgers
q = init_q1(t)

for i in range(t-1):
    ql = q[1:-1,i] + 0.25 * ((q[1:-1,i] - q[:-2,i]) + (q[2:,i]-q[1:-1,i]))
    qr = q[1:-1,i] - 0.25 * ((q[1:-1,i] - q[:-2,i]) + (q[2:,i]-q[1:-1,i]))
    A = 0.5 * (ql[:-1] + qr[1:])
    Fl = 0.5 * ql ** 2
    Fr = 0.5 * qr ** 2
    f = 0.5 * (Fr[1:] + Fl[:-1]) - 0.5 * np.abs(A) * (qr[1:] - ql[:-1])
    q[2:-2,i+1] = q[2:-2,i] - dtdx * ( f[1:] - f[:-1])

plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('Roe-Fromm Burgers')
plt.savefig('HW1/Roefrommburgers.eps', format = 'eps')
# %% Roe Fromm Advection
q = init_q1(t)

for i in range(2):
    ql = q[1:-1,i] + 0.25 * ((q[1:-1,i] - q[:-2,i]) + (q[2:,i]-q[1:-1,i]))
    qr = q[1:-1,i] - 0.25 * ((q[1:-1,i] - q[:-2,i]) + (q[2:,i]-q[1:-1,i]))
    A = 1
    Fl = ql
    Fr = qr
    f = 0.5 * (Fr[1:] + Fl[:-1]) - 0.5 * np.abs(A) * (qr[1:] - ql[:-1])
    q[2:-2,i+1] = q[2:-2,i] - dtdx * ( f[1:] - f[:-1])

plt.ylim(0.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Roe-Fromm advection')
plt.savefig('HW1/Roefrommadvection.eps', format = 'eps')

# %% 2 Task  Godunov advection
q = init_q2(t)
q = godunov_adv(q,t)

plt.ylim(-1.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('Godunov Advection task 2')
plt.savefig('HW1/Godunovadvection2.eps', format = 'eps')

# %% 2 Task  Godunov burgers
q = init_q2(t)
q = godunov_burg(q,t)

plt.ylim(-1.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')
plt.title('Godunov Burgers task 2')
plt.savefig('HW1/Godunovburgers2.eps', format = 'eps')

# %% 2 Task  Roe advection
q = init_q2(t)
q = roe_adv(q,t)

plt.ylim(-1.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('Roe Advection task 2')
plt.savefig('HW1/Roeadvection2.eps', format = 'eps')

# %% 2 Task  Roe burgers
q = init_q2(t)
q = roe_burg(q,t)

plt.ylim(-1.5,2.5)
plt.yticks(np.unique(q[:,1]))
plt.step(q[:,:2],'o-',where='mid')

plt.title('Roe Burgers task 2')
plt.savefig('HW1/Roeburgers2.eps', format = 'eps')
