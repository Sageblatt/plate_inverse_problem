import numpy as np
import pyFreeFem as pyff


MODULI_INDICES = ["11", "12", "16", "22", "26", "66"]

# TODO make it local
tgv = 1e30

def load_matrices_symm(fname: str) -> dict:
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

    script += """
    // Boundary condition u = funcBc on Dirichlet border
    func funcBC = 1; // 1 is default

    fespace Vh(Th, P2Morley);
    Vh [u, ux, uy], [v, vx, vy];

    varf BC([u, ux, uy], [v, vx, vy]) = on(1, u=funcBC, ux=0, uy=0);
    real[int] vBC = BC(0, Vh, tgv=-1);

    // DIRTY HACK: i create a small triangle close to rtest
    // so that even if I fuck up with indices, the difference will not be too large
    real[int] xxtest = [xtest, xtest + 1e-8, xtest + 1e-8];
    real[int] yytest = [ytest, ytest - 1e-8, ytest + 1e-8];
    // Create a surrogate mesh to produce interpolation matrix
    mesh testTh = triangulate(xxtest, yytest);
    // Create a surrogate FE space with P1 basis functions on testTh
    // As P1 basis functions are standard triangular <<caps>> which are equal to 1
    // in nodes, the DOF in this basis are exactly u(x_i, y_i)
    fespace testVh(testTh, P1);
    int[int] u2vc = [0];
    matrix MinterpC = interpolate(testVh, Vh, U2Vc=u2vc);
    """

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

    test_point_coord = np.array([ff_output["xtest"], ff_output["ytest"]]) # Probably not needed

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
        "interpolation_value_from_bc": interpolation_value_from_bc,
        "interpolation_vector": interpolation_vector,
        "test_point_coord": test_point_coord,
        "constrained_idx": constrained_idx,  # free == not(constrained) so returning both is redundant,
        "mesh": ff_output['Th'],
        "boundary_value": f_bc
    }

def load_matrices_unsymm(fname: str):
    with open(fname, "r") as ifile:
        _script = ifile.read()

    script = pyff.edpScript(_script)

    script += """
    real tgv = -1.0;

    fespace Lh(Th, P1);
    Lh u, v, r, s;

    fespace Mh(Th, P2Morley);
    Mh [w, wx, wy], [t, tx, ty];

    // Boundary condition u = funcBc on Dirichlet border
    func funcBC = 1.0;

    varf BCL(u, r) = on(1, u=0);
    varf BCM([w, wx, wy], [t, tx, ty]) = on(1, w=funcBC, wx=0, wy=0);

    varf vmarkLh(u, r) = on(1, u = 1);
    real[int] vmarkerLh = vmarkLh(0, Lh, tgv = -1);

    varf vmarkMh([w, wx, wy], [t, tx, ty]) = on(1, w=100, wx=200, wy=300);
    real[int] vmarkerMh = vmarkMh(0, Mh, tgv = -1);

    real[int] vBCLh = BCL(0, Lh, tgv=-1);
    real[int] vBCMh = BCM(0, Mh, tgv=-1);

    real innermult = 0.3;
    border CAccin(t=0., 2*pi){x=xtest + innermult*rAccel*cos(t); y=ytest + innermult*rAccel*sin(t); label=3;}
    mesh accTh = buildmesh(CAccin(64)); // Cubature formula would be better! Disc: https://doi.org/10.1007/s002110050358 ---------------------------------------------
    fespace midVh(accTh, P1); // More intermediate interpolation steps?----------------------------------------------------------

    int[int] u2vc = [0];
    int[int] u2vc1 = [1];
    int[int] u2vc2 = [2];

    matrix Minterp = interpolate(midVh, Mh, U2Vc=u2vc);
    matrix MinterpWx = interpolate(midVh, Mh, U2Vc=u2vc1);
    matrix MinterpWy = interpolate(midVh, Mh, U2Vc=u2vc2);

    matrix MinterpL = interpolate(midVh, Lh);
    """

    # WARNING: all parts of the Dirichlet BC should be labelled 1
    # ensure it in _problem.edp
    script += pyff.VarfScript(
        Sxx='int2d(Th)(dx(u)*dx(r)) + on(1, u=0)',
        Sxy='int2d(Th)(dx(r)*dy(u)) + on(1, u=0)',
        Syx='int2d(Th)(dx(u)*dy(r)) + on(1, u=0)',
        Syy='int2d(Th)(dy(u)*dy(r)) + on(1, u=0)',
        SxxL='int2d(Th)(dx(u)*dx(r))', # Index L is for K12
        SxyL='int2d(Th)(dx(r)*dy(u))',
        SyxL='int2d(Th)(dx(u)*dy(r))',
        SyyL='int2d(Th)(dy(u)*dy(r))',
        M11='int2d(Th)(u*r) + on(1, u=0)',
        M11Correction='int2d(Th)(indAccel*u*r) + on(1, u=0)',
        functions=['u', 'r'],
        fespaces=['Lh', 'Lh']
    )

    script += pyff.VarfScript(
        Rxxx='int2d(Th)(dx(r)*dxx(w))',
        Rxyy='int2d(Th)(dx(r)*dyy(w))',
        Rxxy='int2d(Th)(dx(r)*dxy(w))',
        Ryxx='int2d(Th)(dy(r)*dxx(w))',
        Ryyy='int2d(Th)(dy(r)*dyy(w))',
        Ryxy='int2d(Th)(dy(r)*dxy(w))',
        M13 ='int2d(Th)(-indAccel*r*dx(w))',
        M23 ='int2d(Th)(-indAccel*r*dy(w))',
        functions=['w', 'r'],
        fespaces=['Mh', 'Lh']
    )

    script += pyff.VarfScript(
        Txxxx='int2d(Th)(dxx(w)*dxx(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Txxyy='int2d(Th)(dxx(w)*dyy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Tyyxx='int2d(Th)(dyy(w)*dxx(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Txxxy='int2d(Th)(dxx(w)*dxy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Txyxx='int2d(Th)(dxy(w)*dxx(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Txyyy='int2d(Th)(dxy(w)*dyy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Tyyxy='int2d(Th)(dyy(w)*dxy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Txyxy='int2d(Th)(dxy(w)*dxy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        Tyyyy='int2d(Th)(dyy(w)*dyy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        M33='int2d(Th)(w*t) + on(1, w=funcBC, wx=0, wy=0)',
        M33Correction='int2d(Th)(indAccel*w*t) + on(1, w=funcBC, wx=0, wy=0)',
        M33I2='int2d(Th)(dx(w)*dx(t) + dy(w)*dy(t)) + on(1, w=funcBC, wx=0, wy=0)',
        M33I2Correction='int2d(Th)(indAccel*(dx(w)*dx(t) + dy(w)*dy(t))) + on(1, w=funcBC, wx=0, wy=0)',
        functions=['[w, wx, wy]', '[t, tx, ty]'],
        fespaces=['Mh', 'Mh']
    )

    # WARNING: keyword 'array' only works with my upgraded pyFreeFem version from the forked repo in my GitHub
    script += pyff.OutputScript(
        vBCLh="array",
        vBCMh="array",
        vmarkerLh="array",
        vmarkerMh="array",
        interp="matrix",
        interpWx="matrix",
        interpWy="matrix",
        interpL="matrix",
        xtest="real",
        ytest="real",
        tgv="real",
        Th='mesh'
    )

    ff_output = script.get_output() # TODO: Refactor code so it doesn't look like ****

    Sxx = ff_output['Sxx'].tocoo()
    Sxy = ff_output['Sxy'].tocoo()
    Syx = ff_output['Syx'].tocoo()
    Syy = ff_output['Syy'].tocoo()

    SxxL = ff_output['SxxL'].tocoo()
    SxyL = ff_output['SxyL'].tocoo()
    SyxL = ff_output['SyxL'].tocoo()
    SyyL = ff_output['SyyL'].tocoo()

    M11 = ff_output['M11']
    M11Correction = ff_output['M11Correction']

    Rxxx = ff_output['Rxxx'].tocoo()
    Rxyy = ff_output['Rxyy'].tocoo()
    Rxxy = ff_output['Rxxy'].tocoo()
    Ryxx = ff_output['Ryxx'].tocoo()
    Ryyy = ff_output['Ryyy'].tocoo()
    Ryxy = ff_output['Ryxy'].tocoo()

    M13 = ff_output['M13'].tocoo()
    M23 = ff_output['M23'].tocoo()

    Txxxx = ff_output['Txxxx'].tocoo()
    Txxyy = ff_output['Txxyy'].tocoo()
    Tyyxx = ff_output['Tyyxx'].tocoo()
    Txxxy = ff_output['Txxxy'].tocoo()
    Txyxx = ff_output['Txyxx'].tocoo()
    Txyyy = ff_output['Txyyy'].tocoo()
    Tyyxy = ff_output['Tyyxy'].tocoo()
    Txyxy = ff_output['Txyxy'].tocoo()
    Tyyyy = ff_output['Tyyyy'].tocoo()

    M33 = ff_output['M33']
    M33Correction = ff_output['M33Correction']

    M33I2 = ff_output['M33I2']
    M33I2Correction = ff_output['M33I2Correction']

    interp_mat = ff_output['interp'].todense()
    interp_mat_Wx = ff_output['interpWx'].todense()
    interp_mat_Wy = ff_output['interpWy'].todense()
    interp_mat_Lh = ff_output['interpL'].todense()

    vBCLh = ff_output['vBCLh'] # is zero, so not used
    vBCMh = ff_output['vBCMh'] # for rhs vector

    Lh_size = vBCLh.size
    Mh_size = vBCMh.size

    all_size = 2*Lh_size + Mh_size

    ns = (all_size, all_size) # new shape -> ns

    marker_Lh = ff_output['vmarkerLh']
    marker_Mh = ff_output['vmarkerMh']

    # indices for rows in matrix
    dLh_idx = np.nonzero(marker_Lh)[0] # Dirichlet BC indices for Lh
    dMh_idx = np.nonzero(marker_Mh)[0] # Dirichlet BC indices for Mh

    def resize(mat):
        m = mat.copy().tocoo()
        m.resize(ns)
        return m

    def move(mat, ncol, nrow):
        mat.col += ncol * Lh_size
        mat.row += nrow * Lh_size

    def rmrows_lh(mat, order):
        if not 0 < order < 3:
            raise ValueError
        rows = dLh_idx + (order - 1) * Lh_size
        rows = rows[:, np.newaxis]
        mask_arr = np.equal(rows, mat.row).sum(axis=0, dtype=bool)
        mat.data[mask_arr] = 0

    def rmrows_mh(mat):
        rows = dMh_idx + 2 * Lh_size
        rows = rows[:, np.newaxis]
        mask_arr = np.equal(rows, mat.row).sum(axis=0, dtype=bool)
        mat.data[mask_arr] = 0

    def transp(mat):
        return (mat + mat.transpose(copy=True)).tocoo()


    KA11 = resize(Sxx)

    KA22 = resize(Syy)
    move(KA22, 1, 1)

    KA12 = resize(SxyL)
    move(KA12, 1, 0)
    KA12 = transp(KA12)
    rmrows_lh(KA12, 1)
    rmrows_lh(KA12, 2)

    KA16K11 = resize(Sxy + Syx)
    KA16K12 = resize(SxxL)
    move(KA16K12, 1, 0)
    KA16K12 = transp(KA16K12)
    rmrows_lh(KA16K12, 1)
    rmrows_lh(KA16K12, 2)
    KA16 = KA16K11 + KA16K12

    KA26K22 = resize(Sxy + Syx)
    move(KA26K22, 1, 1)
    KA26K12 = resize(SyyL)
    move(KA26K12, 1, 0)
    KA26K12 = transp(KA26K12)
    rmrows_lh(KA26K12, 1)
    rmrows_lh(KA26K12, 2)
    KA26 = KA26K12 + KA26K22

    KA66K11 = resize(Syy)
    KA66K22 = resize(Sxx)
    move(KA66K22, 1, 1)
    KA66K12 = resize(SyxL)
    move(KA66K12, 1, 0)
    KA66K12 = transp(KA66K12)
    rmrows_lh(KA66K12, 1)
    rmrows_lh(KA66K12, 2)
    KA66 = KA66K11 + KA66K12 + KA66K22


    KB11 = resize(-Rxxx)
    move(KB11, 2, 0)
    KB11 = transp(KB11)
    rmrows_lh(KB11, 1)
    rmrows_mh(KB11)

    KB22 = resize(-Ryyy)
    move(KB22, 2, 1)
    KB22 = transp(KB22)
    rmrows_lh(KB22, 2)
    rmrows_mh(KB22)

    KB12K13 = resize(-Rxyy)
    move(KB12K13, 2, 0)
    KB12K13 = transp(KB12K13)
    rmrows_lh(KB12K13, 1)
    rmrows_mh(KB12K13)
    KB12K23 = resize(-Ryxx)
    move(KB12K23, 2, 1)
    KB12K23 = transp(KB12K23)
    rmrows_lh(KB12K23, 2)
    rmrows_mh(KB12K23)
    KB12 = KB12K13 + KB12K23

    KB16K13 = resize(-Ryxx)
    move(KB16K13, 2, 0)
    KB16K13 = transp(KB16K13)
    rmrows_lh(KB16K13, 1)
    rmrows_mh(KB16K13)
    KB16K23 = resize(-Rxxx)
    move(KB16K23, 2, 1)
    KB16K23 = transp(KB16K23)
    rmrows_lh(KB16K23, 2)
    rmrows_mh(KB16K23)
    KB16 = KB16K13 + KB16K23

    KB26K13 = resize(-2*Rxxy - Ryyy)
    move(KB26K13, 2, 0)
    KB26K13 = transp(KB26K13)
    rmrows_lh(KB26K13, 1)
    rmrows_mh(KB26K13)
    KB26K23 = resize(-Rxyy - 2*Ryxy)
    move(KB26K23, 2, 1)
    KB26K23 = transp(KB26K23)
    rmrows_lh(KB26K23, 2)
    rmrows_mh(KB26K23)
    KB26 = KB26K13 + KB26K23

    KB66K13 = resize(-2*Ryxy)
    move(KB66K13, 2, 0)
    KB66K13 = transp(KB66K13)
    rmrows_lh(KB66K13, 1)
    rmrows_mh(KB66K13)
    KB66K23 = resize(-2*Rxxy)
    move(KB66K23, 2, 1)
    KB66K23 = transp(KB66K23)
    rmrows_lh(KB66K23, 2)
    rmrows_mh(KB66K23)
    KB66 = KB66K13 + KB66K23


    KD11 = resize(Txxxx)
    move(KD11, 2, 2)

    KD12 = resize(Txxyy + Tyyxx)
    move(KD12, 2, 2)

    KD16 = resize(2*(Txxxy + Txyxx))
    move(KD16, 2, 2)

    KD26 = resize(2*(Txyyy + Tyyxy))
    move(KD26, 2, 2)

    KD66 = resize(4*Txyxy)
    move(KD66, 2, 2)

    KD22 = resize(Tyyyy)
    move(KD22, 2, 2)


    KM11 = resize(M11)
    KM11Corr = resize(M11Correction)

    KM13 = resize(M13)
    move(KM13, 2, 0)
    KM13 = transp(KM13)
    rmrows_lh(KM13, 1)
    rmrows_mh(KM13)

    KM23 = resize(M23)
    move(M23, 2, 1)
    KM23 = transp(KM23)
    rmrows_lh(KM23, 2)
    rmrows_mh(KM23)

    KM22 = resize(M11)
    move(KM22, 1, 1)
    KM22Corr = resize(M11Correction)
    move(KM22Corr, 1, 1)

    KM33 = resize(M33)
    move(KM33, 2, 2)
    KM33Corr = resize(M33Correction)
    move(KM33Corr, 2, 2)
    KM33I2 = resize(M33I2)
    move(KM33I2, 2, 2)
    KM33I2Corr = resize(M33I2Correction)
    move(KM33I2Corr, 2, 2)

    rhs_vec = np.zeros(ns[0], dtype=np.float64)
    rhs_vec[2*Lh_size:] = vBCMh

    return ([KA11, KA12, KA16, KA22, KA26, KA66,
            KB11, KB12, KB16, KB22, KB26, KB66,
            KD11, KD12, KD16, KD22, KD26, KD66,
            KM11, KM11Corr, KM22, KM22Corr, KM33,
            KM33Corr, KM33I2, KM33I2Corr, KM13, KM23],
            rhs_vec, interp_mat, interp_mat_Lh, Lh_size, Mh_size,
            ff_output['Th'], interp_mat_Wx, interp_mat_Wy)
