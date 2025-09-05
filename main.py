import numpy as np
import matplotlib.pyplot as plt


MASS = 1000.      # kg
G = 9.81          # m/s^2
DRAG_C = 5        # N/(m/s)^2

x0, y0 = 0, 10    # m
vx0, vy0 = 2, 3   # m/s
DT = 0.01         # s

init_vel = np.array([vx0, vy0])
init_pos = np.array([x0, y0])

def drag_force(vel):
    vmag = np.linalg.norm(vel)
    return -DRAG_C * vmag * vel


def total_force(vel):
    return drag_force(vel) + np.array([0, -MASS * G])


def energy(pos, vel):
    potential = MASS * G * pos[1]
    kinetic = 1 / 2 * MASS * np.dot(vel, vel)
    total = potential + kinetic
    return np.array(potential, kinetic, total)


def new_step(pos, vel, dt=DT):
    a = total_force(vel) / MASS
    vel_new = vel + a * dt
    pos_new = pos + vel * dt
    return pos_new, vel_new


def simulate(dt=DT):
    pos = np.array([x0, y0])
    vel = np.array([vx0, vy0])
    t = 0
    pe, ke, e = energy(pos, vel)

    t_hist = [0]
    x_hist, y_hist = [pos[0]], [pos[1]]
    vx_hist, vy_hist = [vel[0]], [vel[1]]
    pe_hist, ke_hist, e_hist = [pe], [ke], [e]

    while pos[1] > 0:
        pos, vel = new_step(pos, vel, dt)
        pe, ke, e = energy(pos, vel)
        t += dt

        t_hist.append(t)
        x_hist.append(pos[0])
        y_hist.append(pos[1])
        vx_hist.append(vel[0])
        vy_hist.append(vel[1])
        pe_hist.append(pe)
        ke_hist.append(ke)
        e_hist.append(e)

    return (np.array(t_hist), np.array(x_hist), np.array(y_hist), 
            np.array(vx_hist), np.array(vy_hist), 
            np.array(pe_hist), np.array(ke_hist), np.array(e_hist))






