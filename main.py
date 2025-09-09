import numpy as np
import matplotlib.pyplot as plt


MASS = 1000.      # kg
G = 9.8           # m/s^2
DRAG_C = 5        # N/(m/s)^2

x0, y0 = 0, 10    # m
vx0, vy0 = 2, 3   # m/s
DT = 0.01         # s

init_vel = np.array([vx0, vy0])
init_pos = np.array([x0, y0])

def drag_force(vel):
    vmag = np.linalg.norm(vel)
    return -DRAG_C * vmag * vel


def total_force(pos, vel, mass=MASS, g=G, drag_c=DRAG_C):
    Fg = np.array([0.0, -mass * g])
    vmag = np.linalg.norm(vel)
    Fd = -drag_c * vmag * vel if vmag != 0 else np.zeros(2)
    return Fg + Fd


def energy(pos, vel, mass=MASS, g=G):
    potential = mass * g * max(pos[1], 0.0)
    kinetic = 0.5 * mass * np.dot(vel, vel)
    total = potential + kinetic
    return potential, kinetic, total


def new_step(pos, vel, force, dt=DT, mass=MASS):
    a = force / mass
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
        F = total_force(pos, vel)
        pos_new, vel_new = new_step(pos, vel, F, dt=dt)
        
        if pos[1] > 0 and pos_new[1] <= 0:
            frac = pos[1] / (pos[1] - pos_new[1])
            t += frac * dt
            impact_pos = pos + frac * (pos_new - pos)
            impact_pos[1] = 0.0
            pos, vel = impact_pos, vel + (F / MASS) * (frac * dt)
            break
        pos, vel = pos_new, vel_new
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


t, x, y, vx, vy, pe, ke, e = simulate(DT)
v = np.sqrt(vx**2 + vy**2)

plt.figure()
plt.plot(x, y)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory")
plt.grid(True)

plt.figure()
plt.plot(t, y)
plt.xlabel("t (s)")
plt.ylabel("y (m)")
plt.title("Height vs Time")
plt.grid(True)

plt.figure()
plt.plot(t, vx, label=r"$v_x$")
plt.plot(t, vy, label=r"$v_y$")
plt.plot(t, v, label=r"$v$")
plt.xlabel("t (s)")
plt.ylabel("velocity (m/s)")
plt.title("Velocity vs Time")
plt.legend()
plt.grid(True)

plt.figure()
plt.plot(t, pe, label="PE")
plt.plot(t, ke, label="KE")
plt.plot(t, e, label="Total")
plt.xlabel("t (s)")
plt.ylabel("energy (J)")
plt.title("Energies vs Time")
plt.legend()
plt.grid(True)

plt.show()

