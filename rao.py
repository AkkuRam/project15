import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.io.xarray import problems_from_dataset
import xarray as xr
from capytaine.post_pro.rao import rao
import math
import pyvista as pv


# cpt.set_logging('INFO')

# ----------------------------
# Boat Setup
# ----------------------------

hull = cpt.load_mesh('sree/Hull_cleaned.stl', file_format='stl')
dofs = cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0))
cpt.set_logging('ERROR')

def generate_boat(dofs, hull):
   
    body_hull = cpt.FloatingBody(
        mesh=hull,
        dofs=dofs,
        center_of_mass=(0.0, 0.0, 0.0),
    )
    
    hydro = body_hull.compute_hydrostatics(rho=1025.0)
    body_hull.inertia_matrix = hydro["inertia_matrix"]
    body_hull.hydrostatic_stiffness = hydro["hydrostatic_stiffness"]
    
    return body_hull

# ----------------------------
# BEM Solver
# ----------------------------

def dataset(body):

    test_matrix = xr.Dataset(coords={
        "omega": [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        "wave_direction": [math.pi/12, math.pi/6, math.pi/2, math.pi],
        "radiating_dof": list(body.dofs),
        "water_depth": 200,
        "rho": 1025
    })

    solver = cpt.BEMSolver()
    pbs = problems_from_dataset(test_matrix, body)
    results = solver.solve_all(pbs, keep_details=True)
    ds = cpt.assemble_dataset(results)
    ds["RAO"] = rao(ds)

    return ds, results, solver


# ----------------------------
# Running methods
# ----------------------------

body_hull = generate_boat(dofs, hull)
body_hull.show()
ds, results, solver = dataset(body_hull)



# ----------------------------
# Saving RAO Results
# ----------------------------

def split_complex_vars(ds):
    out = xr.Dataset()
    for name, var in ds.data_vars.items():
        if np.iscomplexobj(var):
            out[name + "_real"] = np.real(var)
            out[name + "_imag"] = np.imag(var)
        else:
            out[name] = var
    for name, coord in ds.coords.items():
        out = out.assign_coords({name: coord})
    return out


ds_split = split_complex_vars(ds)

# Fix DOF coords on ds_split, not ds
for coord in ds_split.coords:
    if "dof" in coord:
        ds_split[coord] = ds_split[coord].astype(str)

print(ds_split)

ds_split.to_netcdf(
    path="results.nc",
    mode="w",
    engine="netcdf4"
)

 

