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

def drag_force(vel, drag_c=DRAG_C):
    vmag = np.linalg.norm(vel)
    if vmag == 0.0 or drag_c == 0.0:
        return np.zeros(2)
    return -drag_c * vmag * vel


def total_force(pos, vel, mass=MASS, g=G, drag_c=DRAG_C):
    Fg = np.array([0.0, -mass * g])
    Fd = drag_force(vel, drag_c)
    return Fg + Fd


def energy(pos, vel, mass=MASS, g=G):
    y = max(float(pos[1]), 0.0)
    potential = mass * g * y
    kinetic = 0.5 * mass * float(np.dot(vel, vel))
    total = potential + kinetic
    return potential, kinetic, total


def new_step(pos, vel, force, dt, mass=MASS): #semi-implicit Euler
    a = force / mass
    vel_new = vel + a * dt
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


def simulate(dt=DT):
    pos = np.array([x0, y0], dtype=float)
    vel = np.array([vx0, vy0], dtype=float)
    t = 0.0
   

    t_hist = [t]
    x_hist = [pos[0]]
    y_hist = [pos[1]]
    vx_hist = [vel[0]]
    vy_hist = [vel[1]]
    pe, ke, e = energy(pos, vel)
    pe_hist, ke_hist, e_hist = [pe], [ke], [e]

    while pos[1] > 0.0:
        F = total_force(pos, vel)
        pos_new, vel_new = new_step(pos, vel, F, dt)
        t_new = t + dt
        
        if pos[1] > 0.0 and pos_new[1] <= 0.0:
            frac = pos[1] / (pos[1] - pos_new[1])
            frac = max(0.0, min(1.0, frac))
            t_imp = t + frac * dt
            imp_pos = pos + (pos_new - pos) * frac
            imp_pos[1] = 0.0
            a = F / MASS
            imp_vel = vel + a * (frac * dt)

            t_hist.append(t_imp)
            x_hist.append(imp_pos[0]); y_hist.append(imp_pos[1])
            vx_hist.append(imp_vel[0]); vy_hist.append(imp_vel[1])
            pe, ke, e = energy(imp_pos, imp_vel)
            pe_hist.append(pe); ke_hist.append(ke); e_hist.append(e)
            break
        pos, vel, t = pos_new, vel_new, t_new

        t_hist.append(t)
        x_hist.append(pos[0]); y_hist.append(pos[1])
        vx_hist.append(vel[0]); vy_hist.append(vel[1])
        pe, ke, e = energy(pos, vel)
        pe_hist.append(pe); ke_hist.append(ke); e_hist.append(e)

    return (np.array(t_hist), np.array(x_hist), np.array(y_hist), 
            np.array(vx_hist), np.array(vy_hist), 
            np.array(pe_hist), np.array(ke_hist), np.array(e_hist))

if __name__ == "__main__":
    t, x, y, vx, vy, pe, ke, e = simulate(DT)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory")
    plt.grid(True)
    
    v = np.sqrt(vx**2 + vy**2)

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

