import numpy as np
import pyFreeFem as pyff


def getOutput(fname: str):
    # load file with preliminaries related to FreeFem
    # mesh generation/input and definition of test point(s)
    # is here
    with open("_problem.edp", "r") as ifile:
        _script = ifile.read()
    script = pyff.edpScript(_script)

    # WARNING: all parts of the Dirichlet BC should be labelled 1
    # ensure it in _problem.edp
    script += pyff.VarfScript(
        # Stiffness terms depending on each of the anisotropic coefs
        K11="int2d(Th)(dxx(u)*dxx(v)) + on(1, u=0, ux=0, uy=0)",
        K12="int2d(Th)(dyy(u)*dxx(v) + dxx(u)*dyy(v)) + on(1, u=0, ux=0, uy=0)",
        K16="int2d(Th)(dxy(u)*dxx(v) + 2.*dxx(u)*dxy(v)) + on(1, u=0, ux=0, uy=0)",
        K22="int2d(Th)(dyy(u)*dyy(v)) + on(1, u=0, ux=0, uy=0)",
        K26="int2d(Th)(dxy(u)*dyy(v) + 2.*dyy(u)*dxy(v)) + on(1, u=0, ux=0, uy=0)",
        K66="int2d(Th)(2.*dxy(u)*dxy(v)) + on(1, u=0, ux=0, uy=0)",
        # Rotational inertia term
        L="int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v)) + on(1, u=0, ux=0, uy=0)",
        # Mass matrix
        M="int2d(Th)(u*v) + on(1, u=0, ux=0, uy=0)",
        # For correct usage of vectorial element
        functions=["[u, ux, uy]", "[v, vx, vy]"],
    )

    # WARNING: keyword 'array' only works with my upgraded FreeFem version from the forked repo in my GitHub
    script += pyff.OutputScript(
        vLoad="array", vBC="array", interpC="matrix", xtest="real", ytest="real",
    )
    return script.get_output()

def processFFOutput(ff_output: dict):
    return (1, )*6 # stub
