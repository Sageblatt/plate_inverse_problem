import numpy as np
import pyFreeFem as pyff


MODULI_INDICES = ["11", "12", "16", "22", "26", "66"]

# TODO make it local
tgv = 1e30

def load_matrices_symm(fname: str):
    """Processes output of FreeFEM, returns matrices ready to be used in jax model
        with rhs defined by Dirichlet BC. (Midplane-symmetrical plates)

        :param ff_output: result of FreeFem execution
        :type ff_output: dict
    """
    # load file with preliminaries related to FreeFem
    # mesh generation/input and definition of test point(s)
    # is here
    with open(fname, "r") as ifile:
        _script = ifile.read()
    script = pyff.edpScript(_script)

    # WARNING: all parts of the Dirichlet BC should be labelled 1
    # ensure it in _problem.edp
    script += pyff.VarfScript(
        # Stiffness terms depending on each of the anisotropic coefs
        K11="int2d(Th)(dxx(u)*dxx(v)) + on(1, u=0, ux=0, uy=0)",
        K12="int2d(Th)(dyy(u)*dxx(v) + dxx(u)*dyy(v)) + on(1, u=0, ux=0, uy=0)",
        K16="int2d(Th)(2.*dxy(u)*dxx(v) + 2.*dxx(u)*dxy(v)) + on(1, u=0, ux=0, uy=0)",
        K22="int2d(Th)(dyy(u)*dyy(v)) + on(1, u=0, ux=0, uy=0)",
        K26="int2d(Th)(2.*dxy(u)*dyy(v) + 2.*dyy(u)*dxy(v)) + on(1, u=0, ux=0, uy=0)",
        K66="int2d(Th)(4.*dxy(u)*dxy(v)) + on(1, u=0, ux=0, uy=0)",
        # Rotational inertia term
        L="int2d(Th)(dx(u)*dx(v) + dy(u)*dy(v)) + on(1, u=0, ux=0, uy=0)",
        # Mass matrix
        M="int2d(Th)(u*v) + on(1, u=0, ux=0, uy=0)",
        # Rotational inertia term
        LCorrection="int2d(Th)(indAccel*(dx(u)*dx(v) + dy(u)*dy(v))) + on(1, u=0, ux=0, uy=0)",
        # Mass matrix
        MCorrection="int2d(Th)(indAccel*(u*v)) + on(1, u=0, ux=0, uy=0)",
        # For correct usage of vectorial element
        functions=["[u, ux, uy]", "[v, vx, vy]"],
    )

    # WARNING: keyword 'array' only works with my upgraded pyFreeFem version from the forked repo in my GitHub
    script += pyff.OutputScript(
        vLoad="array",
        vBC="array",
        interpC="matrix",
        xtest="real",
        ytest="real",
        tgv="real",
        Th='mesh'
    )

    ff_output = script.get_output()

    def evaluateMatrixAndRHS(K_total, f_bc: np.ndarray, return_indices=False):
        """Processes system of equation for each parameter

            In FreeFem++, system K is constructed for both constrained nodes (value known from Dirichlet BC) and free nodes.
            If k is constrained, and K is a matrix of some variational term V, then K[k, k] == tgv = 1e30
            f_bc[k] == value defined by Dirichlet BC (in my implementation, see _problem.edp)
            We would like to construct the system only for free nodes to decrease the dimension, so we clip K -> F[free_idx, free_idx],
            and construct rhs [f_bc]_i = -\sum_k g_k*V(phi_i, phi_k) for all k in constrained nodes

            :param K_total: Matrix from FreeFem
            :type K_total: `scipy.sparse.spmatrix`
            :param K_total: vector with from FreeFem
            :type K_total: `numpy.ndarray`

            :return: A tuple (K, f)
            :rtype: tuple
            """
        K_total = K_total.todense()

        # Separating free and constrained nodes
        diagK = np.diagonal(K_total)
        free_idx = diagK < tgv
        constrained_idx = diagK >= tgv

        K = K_total[free_idx, :][:, free_idx]

        bcCoefs = K_total[constrained_idx, :][:, free_idx]
        bcValues = f_bc[constrained_idx]

        # This is a short version of formula
        # [f_bc]_i = -\sum_k g_k*V(phi_i, phi_k) for all k in constrained nodes
        rhs = -(bcValues * bcCoefs).sum(axis=0)
        rhs = np.ravel(rhs)
        if return_indices:
            return K, rhs, free_idx, constrained_idx
        else:
            return K, rhs

    # TODO get rid of global variable
    tgv = ff_output["tgv"]
    f_bc = ff_output["vBC"]

    M, fM, free_idx, constrained_idx = evaluateMatrixAndRHS(
        ff_output["M"], f_bc, return_indices=True
    )
    L, fL = evaluateMatrixAndRHS(ff_output["L"], f_bc)

    MCorrection, fMCorrection = evaluateMatrixAndRHS(ff_output["MCorrection"], f_bc)
    LCorrection, fLCorrection = evaluateMatrixAndRHS(ff_output["LCorrection"], f_bc)

    dim = M.shape[0]
    Ks = np.zeros((6, dim, dim))
    fKs = np.zeros((6, dim))
    for idx, ss in enumerate(MODULI_INDICES):
        key = "K" + ss
        K, f = evaluateMatrixAndRHS(ff_output[key], f_bc)
        Ks[idx, :, :] = K
        fKs[idx, :] = f

    interpolation_vector = np.ravel(ff_output["interpC"][0, :].todense())

    interpolation_value_from_bc = (
        interpolation_vector[constrained_idx] @ f_bc[constrained_idx]
    )
    interpolation_vector = interpolation_vector[free_idx]

    f_load = np.ravel(ff_output["vLoad"])[free_idx]
    test_point_coord = np.array([ff_output["xtest"], ff_output["ytest"]])

    return {
        "Ks": Ks,
        "fKs": fKs,
        "M": M,
        "fM": fM,
        "L": L,
        "fL": fL,
        "MCorrection": MCorrection,
        "fMCorrection": fMCorrection,
        "LCorrection": LCorrection,
        "fLCorrection": fLCorrection,
        "fLoad": f_load,
        "interpolation_value_from_bc": interpolation_value_from_bc,
        "interpolation_vector": interpolation_vector,
        "test_point_coord": test_point_coord,
        "constrained_idx": constrained_idx,  # free == not(constrained) so returning both is redundant,
        "mesh": ff_output['Th'],
        "boundary_value": f_bc
    }
