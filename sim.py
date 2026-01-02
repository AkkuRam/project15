import numpy as np
import pyvista as pv

# -------------------------
# Wave spectrum + wave field
# -------------------------
def jonswap_Sf(f_hz, Hs=2.0, Tp=8.0, gamma=3.3):
    fp = 1.0 / Tp
    f = np.asarray(f_hz)

    S = np.zeros_like(f, dtype=float)
    mask = f > 0
    f2 = f[mask]

    sigma = np.where(f2 <= fp, 0.07, 0.09)
    r = np.exp(-0.5 * ((f2 / fp - 1.0) / sigma) ** 2)
    shape = (f2 ** -5.0) * np.exp(-(5.0 / 4.0) * (fp / f2) ** 4.0) * (gamma ** r)

    target_m0 = (Hs / 4.0) ** 2
    m0_tmp = np.trapz(shape, f2)
    scale = target_m0 / m0_tmp if m0_tmp > 0 else 0.0

    S[mask] = scale * shape
    return S

def make_components(Hs=2.0, Tp=8.0, gamma=3.3, fmin=0.03, fmax=1.0, nfreq=128, seed=1):
    rng = np.random.default_rng(seed)
    f = np.linspace(fmin, fmax, nfreq)
    df = f[1] - f[0]
    S = jonswap_Sf(f, Hs=Hs, Tp=Tp, gamma=gamma)

    a = np.sqrt(2.0 * S * df)
    omega = 2.0 * np.pi * f
    g = 9.81
    k = (omega**2) / g
    phi = rng.uniform(0, 2*np.pi, size=nfreq)
    return a, k, omega, phi

def eta_field(xx, yy, t, a, k, omega, phi, theta):
    proj = xx[None, :, :] * np.cos(theta)[:, None, None] + yy[None, :, :] * np.sin(theta)[:, None, None]
    return np.sum(
        a[:, None, None] * np.cos(k[:, None, None] * proj - omega[:, None, None] * t + phi[:, None, None]),
        axis=0
    )

# -------------------------
# Rotation helpers
# -------------------------
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# -------------------------
# Boat helpers
# -------------------------
def scale_mesh_to_length(mesh: pv.PolyData, target_length_m: float) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    ext_x = xmax - xmin
    ext_y = ymax - ymin
    current_len = max(ext_x, ext_y)
    if current_len <= 0:
        return 1.0
    s = target_length_m / current_len
    mesh.scale([s, s, s], inplace=True)
    return s

# -------------------------
# Animation parameters
# -------------------------
Hs, Tp, gamma = 2.0, 8.0, 3.3
Lx, Ly = 300.0, 250.0
nx, ny = 75, 38

fps = 60
dt = 1.0 / fps
T  = 50.0

speed = 10.0
spread_deg = 25.0

# Boat controls
boat_file = "sree/Ship.stl"
target_boat_length_m = 30.0

# IMPORTANT: this is keel depth below the waterline.
# Set to 0.0 for "touching the water surface".
draft_m = 0.0

stl_in_mm = True

# If STL orientation is wrong, set these (degrees)
pre_rot_x_deg = 0.0
pre_rot_y_deg = 0.0
pre_rot_z_deg = 0.0

# Boat horizontal position
cx, cy = 80.0, 0.0

# -------------------------
# Build wave grid
# -------------------------
x = np.linspace(0.0, Lx, nx)
y = np.linspace(-Ly/2.0, Ly/2.0, ny)
xx, yy = np.meshgrid(x, y, indexing="ij")
surface = pv.StructuredGrid(xx, yy, np.zeros_like(xx))

a, k, omega, phi = make_components(Hs=Hs, Tp=Tp, gamma=gamma, nfreq=128, seed=2)
rng_theta = np.random.default_rng(123)
theta = np.deg2rad(rng_theta.uniform(-spread_deg, spread_deg, size=len(a)))

ix = int(np.argmin(np.abs(x - cx)))
iy = int(np.argmin(np.abs(y - cy)))

# Waves travel mainly +X => incoming from -X
yaw_base = np.arctan2(0.0, -1.0)  # = pi, faces -X

# -------------------------
# Load + prep boat STL
# -------------------------
boat0 = pv.read(boat_file).triangulate()
boat0.translate(-np.array(boat0.center), inplace=True)
boat0.rotate_z(90, inplace=True)

if stl_in_mm:
    boat0.scale([0.001, 0.001, 0.001], inplace=True)

if pre_rot_x_deg != 0:
    boat0.rotate_x(pre_rot_x_deg, inplace=True)
if pre_rot_y_deg != 0:
    boat0.rotate_y(pre_rot_y_deg, inplace=True)
if pre_rot_z_deg != 0:
    boat0.rotate_z(pre_rot_z_deg, inplace=True)

scale_mesh_to_length(boat0, target_boat_length_m)

boat0_pts = boat0.points.copy()
boat = boat0.copy(deep=True)

# -------------------------
# Plotter
# -------------------------
p = pv.Plotter()
p.add_mesh(surface, opacity=0.85, color="royalblue")
p.add_axes()
p.camera_position = "xz"
p.add_mesh(boat, color="orange", smooth_shading=True)
p.show(auto_close=False, interactive_update=True)

# -------------------------
# Animation loop
# -------------------------
t = 0.0
nsteps = int(T / dt)

for _ in range(nsteps):
    zz = eta_field(xx, yy, t, a, k, omega, phi, theta)
    surface.points[:, 2] = zz.ravel(order="F")

    # water height & slopes at boat location
    z_water = zz[ix, iy]
    dzz_dx, dzz_dy = np.gradient(zz, x, y, edge_order=1)
    sx = dzz_dx[ix, iy]
    sy = dzz_dy[ix, iy]

    pitch = -np.arctan(sx)
    roll  =  np.arctan(sy)
    yaw   = yaw_base

    # Rotation matrix
    R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)

    # Rotate points (still centered around origin)
    pts_rot = boat0_pts @ R.T

    # --- KEY FIX ---
    # Find the CURRENT lowest point after rotation, then lift/drop so it meets waterline
    zmin_rot = pts_rot[:, 2].min()
    z_translate = (z_water - draft_m) - zmin_rot

    pts_world = pts_rot + np.array([cx, cy, z_translate])

    boat.points = pts_world
    boat.Modified()

    p.update()
    t += dt * speed

p.close()
