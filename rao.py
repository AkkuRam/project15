import numpy as np
import capytaine as cpt
from capytaine.bem.airy_waves import airy_waves_free_surface_elevation
from capytaine.io.xarray import problems_from_dataset
from capytaine.ui.vtk import Animation
from capytaine.meshes.predefined.rectangles import mesh_rectangle
import xarray as xr
from capytaine.post_pro.rao import rao
import math

# cpt.set_logging('INFO')

mesh = cpt.mesh_horizontal_cylinder(
    length=2.75, radius=0.5,  
    center=(0, 0, 0),         
    resolution=(5, 20, 40),
    ).immersed_part()    

lid_mesh = cpt.mesh_rectangle(
    size=(1, 2.75),       
    center=(0, 0, 0),
    faces_max_radius=0.05,     
    normal=(0.0, 0.0, -1.0),        
)

body = cpt.FloatingBody(mesh=mesh,lid_mesh=lid_mesh, dofs=cpt.rigid_body_dofs(rotation_center=(0.000000, 0.000000, 0.000000)), 
                        center_of_mass=(0.000000, 0.000000, 0.000000))
hydro = body.compute_hydrostatics(rho=1025.0)  
body.inertia_matrix = hydro["inertia_matrix"]
body.hydrostatic_stiffness = hydro["hydrostatic_stiffness"]
body.show()

test_matrix = xr.Dataset(coords={
    "omega": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5],
    "wave_direction": [0, math.pi/12, math.pi/6, math.pi/2, math.pi],
    "radiating_dof": list(body.dofs),
    "water_depth": 200,
    "rho": 1025
})

solver = cpt.BEMSolver()
pbs = problems_from_dataset(test_matrix, body)
results = solver.solve_all(pbs, keep_details=True)
ds = cpt.assemble_dataset(results)

ds["RAO"] = rao(ds)

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

for coord in ds.coords:
    if "dof" in coord:
        ds[coord] = ds[coord].astype(str)
        
print(ds)
ds.to_netcdf(path="results.nc", mode="w", auto_complex="true")
 

