import pyvista as pv
import numpy as np
import pandas as pd

# ----------------------------
# DOFs data for sea state 4
# ----------------------------

response_df = pd.read_csv('rao_response_spectrum.csv')
  
rao_pitch = response_df["Pitch"].to_numpy()    
rao_roll  = response_df["Roll"].to_numpy()      
rao_yaw   = response_df["Yaw"].to_numpy()

rao_heave = response_df["Heave"].to_numpy() 
rao_sway  = response_df["Sway"].to_numpy()     
rao_surge = response_df["Surge"].to_numpy()

rao_pitch = np.ones_like(rao_pitch)   
rao_roll  = np.ones_like(rao_roll)  
rao_yaw   = np.ones_like(rao_yaw)    

rao_heave = np.ones_like(rao_heave)
rao_sway  = np.ones_like(rao_sway)  
rao_surge = np.ones_like(rao_surge) 

# ----------------------------
# JONSWAP Spectrum
# ----------------------------

g = 9.81
Hs = 2.51
Tp = 8.33
gamma = 3.3

f = np.linspace(0.01, 1.0, 50)
omega = 2 * np.pi * f
fp = 1.0 / Tp
sigma = np.where(f <= fp, 0.07, 0.09)
a = 0.0081

def jonswap_spectrum(f, fp, a, gamma, sigma, g=9.81):
    return (
        a * g**2 * (2*np.pi)**-4 * f**-5
        * np.exp(-1.25 * (fp / f)**4)
        * gamma ** np.exp(-0.5 * ((f - fp) / (sigma * fp))**2)
    )

for _ in range(3): 
    S_jonswap = jonswap_spectrum(f, fp, a, gamma, sigma, g)
    m0 = np.trapezoid(S_jonswap, f)
    Hs_new = 4.0 * np.sqrt(m0)
    a *= (Hs / Hs_new)**2

S_jonswap = jonswap_spectrum(f, fp, a, gamma, sigma, g)
k = omega**2 / g

np.random.seed(42)
phases = np.random.uniform(0, 2*np.pi, len(f))
directions = np.random.uniform(0, 2*np.pi, len(f))

# ----------------------------
# Rendering JONSWAP Spectrum
# ----------------------------

def update_waves(t, x_grid, y_grid, f):
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
# Create wave grid
# ----------------------------
pl = pv.Plotter()

def wave_grid(pl, start, end, points):
    x = np.linspace(start, end, points)
    y = np.linspace(start, end, points)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros_like(x_grid)
    x_grid, y_grid = np.meshgrid(x, y)

    points = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [len(x), len(y), 1]

    mesh = pl.add_mesh(
        grid,
        scalars=grid.points[:, 2],
        show_edges=False,
        specular=0.5,
    )

    return mesh, grid, x_grid, y_grid, z_grid

# ----------------------------
# Boat mesh
# ----------------------------

boat_mesh = pv.read("sree/Ship.stl").triangulate()
xmin, xmax, *_ = boat_mesh.bounds
current_length = xmax - xmin
target_length = 2.75
scale_factor = target_length / current_length

boat_mesh.scale([scale_factor]*3, inplace=True)
boat = boat_mesh.copy()
pl.add_mesh(boat, color='tan', specular=0.5, smooth_shading=True)

# ----------------------------
# Animation
# ----------------------------
t = 0
dt = 0.1
df = np.gradient(f)
mesh, grid, x_grid, y_grid, z_grid = wave_grid(pl, -20, 20, 60)
wave_amps = np.sqrt(2 * S_jonswap * df)

def callback(step):
    global t
    t += dt

    z = update_waves(t, x_grid, y_grid, f)
    grid.points[:, 2] = z.ravel()

    pitch = np.sum(rao_pitch * wave_amps * np.cos(omega*t + phases))
    roll  = np.sum(rao_roll  * wave_amps * np.cos(omega*t + phases))
    yaw   = np.sum(rao_yaw   * wave_amps * np.cos(omega*t + phases))

    heave = np.sum(rao_heave * wave_amps * np.cos(omega*t + phases))
    surge = np.sum(rao_surge * wave_amps * np.cos(omega*t + phases))
    sway  = np.sum(rao_sway  * wave_amps * np.cos(omega*t + phases))

    boat.points[:] = boat_mesh.points
    boat.translate([surge, sway, heave], inplace=True)
    boat.rotate_x(roll, point=boat.center, inplace=True)
    boat.rotate_y(pitch, point=boat.center, inplace=True)
    boat.rotate_z(yaw, point=boat.center, inplace=True)

    pl.render()

pl.camera.position = (65, -40, 30)
pl.enable_trackball_style()   
pl.add_timer_event(max_steps=500, duration=50, callback=callback)
pl.show()