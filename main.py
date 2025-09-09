import numpy as np
import matplotlib.pyplot as plt


MASS = 1000.      # kg
G = 9.8           # m/s^2
DRAG_C = 0        # N/(m/s)^2

x0, y0 = 0, 1000    # m
vx0, vy0 = 1, 100   # m/s
DT = 0.001         # s

init_vel = np.array([vx0, vy0])  # set initial conditions
init_pos = np.array([x0, y0])

def drag_force(vel, c=DRAG_C):             # drag_force = -c*|v|*v
    vmag = np.linalg.norm(vel)
    return -c * vmag * vel


def total_force(vel, c=DRAG_C):
    return drag_force(vel, c) + np.array([0, -MASS * G])


def energy(pos, vel):
    potential = MASS * G * pos[1]
    kinetic = 1 / 2 * MASS * np.dot(vel, vel)
    total = potential + kinetic
    return potential, kinetic, total


def new_step(pos, vel, c=DRAG_C, dt=DT):
    a = total_force(vel, c) / MASS
    vel_new = vel + a * dt
    pos_new = pos + vel * dt
    return pos_new, vel_new


def simulate(c=DRAG_C, dt=DT):
    pos = np.array([x0, y0])
    vel = np.array([vx0, vy0])
    t = 0
    pe, ke, e = energy(pos, vel)

    t_hist = [0]                                # lists for history of variables
    x_hist, y_hist = [pos[0]], [pos[1]]
    vx_hist, vy_hist = [vel[0]], [vel[1]]
    pe_hist, ke_hist, e_hist = [pe], [ke], [e]

    while pos[1] > 0:                           # simulation loop
        pos, vel = new_step(pos, vel, c, dt)
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


"""Start simulation and make plots"""

# simulate with the preset drag constant and time step
t, x, y, vx, vy, pe, ke, e = simulate(c=0, dt=DT)

# Write output file
with open("trajectory_output.txt", "w") as f:
    for ti, xi, yi in zip(t, x, y):
        f.write(f"{ti} {xi} {yi}\n")

print("Output written to trajectory_output.txt")
v = np.sqrt(vx**2 + vy**2)

plt.figure()
plt.plot(x, y)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory with time step 0.01")
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


"""Addressing the questions"""
"""

# 7 i.
print("Initial energy:", e[0]) 
print("Final energy:", e[-1])
print("Percentage of the change of energy:", (e[0] - e[-1])/e[0])


# 7 ii.
# simulate with time step=1
t1, x1, y1, vx1, vy1, pe1, ke1, e1 = simulate(DRAG_C, 1)
# simulate with time step=0.0001
t2, x2, y2, vx2, vy2, pe2, ke2, e2 = simulate(DRAG_C, 0.0001)

plt.figure()
plt.plot(x1, y1)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory with time step 1")
plt.grid(True)

plt.figure()
plt.plot(x2, y2)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory with time step 0.0001")
plt.grid(True)

plt.show()


# 7 iii.
# simulate with drag constant = 0
t3, x3, y3, vx3, vy3, pe3, ke3, e3 = simulate(0, 0.0001)

plt.figure()
plt.plot(x3, y3)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectory with no air resistance")
plt.grid(True)

plt.show()

print("The landing point without air resistance is:", (float(x[-1]), float(y[-1])))

"""
