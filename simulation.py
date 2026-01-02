
import pyvista as pv
import numpy as np
import pandas as pd

response_df = pd.read_csv('rao_response_spectrum.csv')
rao_heave = response_df["Heave"].to_numpy()     
rao_roll  = response_df["Roll"].to_numpy()      
rao_pitch = response_df["Pitch"].to_numpy()  

rao_heave = np.ones_like(rao_heave) * 2.0     
rao_roll  = np.ones_like(rao_roll)  * 10.0   
rao_pitch = np.ones_like(rao_pitch) * 8.0  

# ----------------------------
# Create plotter
# ----------------------------
plotter = pv.Plotter()
plotter.enable_trackball_style()  # Enable interactive camera

# ----------------------------
# Create smaller mesh grid for ocean surface
# ----------------------------
x = np.linspace(-15, 15, 60)
y = np.linspace(-15, 15, 60)
x_grid, y_grid = np.meshgrid(x, y)

points = np.c_[x_grid.ravel(), y_grid.ravel(), np.zeros(x_grid.size)]
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [len(x), len(y), 1]

mesh = plotter.add_mesh(
    grid,
    scalars=grid.points[:, 2],
    show_edges=False,
    specular=0.5,
    cmap='ocean'
)

# ----------------------------
# JONSWAP spectrum parameters
# ----------------------------
g = 9.81
Hs = 2.5
Tp = 6.0    
gamma = 3.3

f = np.linspace(0.01, 1.0, 50)
omega = 2 * np.pi * f

fp = 1.0 / Tp
alpha = 0.0081
sigma = np.where(f <= fp, 0.07, 0.09)

S_jonswap = (
    alpha * g**2 / (2*np.pi)**4 / f**5 *
    np.exp(-1.25 * (fp / f)**4) *
    gamma ** np.exp(-0.5 * ((f - fp) / (sigma * fp))**2)
)

np.random.seed(42)
phases = np.random.uniform(0, 2*np.pi, len(f))
directions = np.random.uniform(0, 2*np.pi, len(f))
k = omega**2 / g

# ----------------------------
# Animation parameters
# ----------------------------
t = 0.0
dt = 0.1

def update_waves(t):
    """Update wave surface based on JONSWAP spectrum"""
    z = np.zeros_like(x_grid)
    for i in range(len(f)):
        df = f[i+1] - f[i] if i < len(f) - 1 else f[i] - f[i-1]
        amp = np.sqrt(2 * S_jonswap[i] * df)

        z += amp * np.cos(
            k[i] * (x_grid * np.cos(directions[i]) + y_grid * np.sin(directions[i]))
            - omega[i] * t + phases[i]
        )
    return z

# ----------------------------
# Load and place the boat (centered)
# ----------------------------
boat_ref = pv.read("sree/Ship.stl").triangulate()
boat_ref.scale([0.002, 0.002, 0.002], inplace=True)
boat_ref.rotate_z(90, point=boat_ref.center, inplace=True)

# move horizontally to wave center
target_center = np.array(grid.center)
shift_xy = target_center - np.array(boat_ref.center)
shift_xy[2] = 0.0
boat_ref.translate(shift_xy, inplace=True)

# -----------------------------
# CRITICAL FIX: WATERLINE ALIGN
# -----------------------------
zmin = boat_ref.bounds[4]      # lowest point of hull
waterline_z = 0.0              # wave mean level
draft = 0.3                    # meters below waterline (tune this)

boat_ref.translate([0, 0, waterline_z - zmin - draft], inplace=True)

boat = boat_ref.copy()
boat_actor = plotter.add_mesh(
    boat,
    color='tan',
    opacity=0.9,
    smooth_shading=True
)


# ----------------------------
# Animation callback
# ----------------------------

df = np.gradient(f)
wave_amps = np.sqrt(2 * S_jonswap * df)
def callback(step):
    global t
    t += dt

    # --- Update waves ---
    z = update_waves(t)
    grid.points[:, 2] = z.ravel()

    # --- RAO-based motions ---
    heave = np.sum(rao_heave * wave_amps * np.cos(omega*t + phases))
    roll  = np.sum(rao_roll  * wave_amps * np.cos(omega*t + phases))
    pitch = np.sum(rao_pitch * wave_amps * np.cos(omega*t + phases))

    # --- Reset boat to reference ---
    boat.points[:] = boat_ref.points

    # heave
    boat.translate([0, 0, heave], inplace=True)

    # roll and pitch 
    boat.rotate_x(roll, point=boat.center, inplace=True)
    boat.rotate_y(pitch, point=boat.center, inplace=True)

    plotter.render()



# ----------------------------
# Run animation
# ----------------------------
plotter.add_timer_event(max_steps=500, duration=50, callback=callback)
plotter.show()
