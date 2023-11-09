import os
import json
import numpy as np
from .Utils import get_jax_plate_dir
from .ParamTransforms import isotropic, orthotropic, orthotropic_d4


ATYPES = {'isotropic': ['E', 'G', 'beta'],
          'orthotropic': ['E1', 'E2', 'G12', 'nu12', 'beta'],
          'orthotropic_d4': ['E1', 'E2', 'G12', 'nu12',
                             'b1', 'b2', 'b3', 'b4']}


class MaterialParams:
    """Class that represents the list of parameters for given material."""
    def __init__(self, density: float, atype: str, **atype_kwargs):
        self.density = density
        self.atype = atype
        if atype not in ATYPES:
            raise ValueError(f'Invalid anisotropy type {atype} for material. '
                             f'Supported options are: {list(ATYPES.keys())}.')

        for param in atype_kwargs:
            if param in ATYPES[atype]:
                setattr(self, param, atype_kwargs[param])

            else:
                raise ValueError(f'Invalid parameter {param} for anisotropy '
                                 f'type {atype}.')


class Material:
    """
    Class that represents material and it's properties, provides interface to
    *.json files.
    """
    def __init__(self, name_or_params: str | MaterialParams):
        """
        Constructor method.

        Parameters
        ----------
        name_or_params : str | MaterialParams
            Name of the material to search for in `materials` folder or
            a dataclass with corresponding values.

        Returns
        -------
        None

        """
        params = None

        if isinstance(name_or_params, str):
            fpath = os.path.join(get_jax_plate_dir(), 'materials',
                                 name_or_params + '.json')

            if os.path.exists(fpath):
                with open(fpath, 'r') as file:
                    params = json.load(file)
                    try:
                        kwargs = {k:v for k, v in params.items() if k not in ['density', 'atype']}
                        params = MaterialParams(params['density'],
                                                params['atype'],
                                                **kwargs)

                    except KeyError as err:
                        raise RuntimeError('One of required parameters was '
                                           'not provided by the .json file '
                                           f'{fpath}.') from err

            else:
                raise ValueError(f'Could not find file {name_or_params}.json '
                                 'in `materials` folder.')

        elif isinstance(name_or_params, MaterialParams):
            params = name_or_params

        else:
            raise TypeError('Argument `name_or_params` should have type '
                            '`str` or `MaterialParams.`')

        self.density = params.density
        self.atype = params.atype
        for param in ATYPES[self.atype]:
            setattr(self, param, getattr(params, param, None))

        if self.atype == 'isotropic':
            self.transform = isotropic

            def get_Ds(h: float, E: float, G: float) -> tuple[float, float]:
                nu = E / (2.0 * G) - 1.0
                D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
                return D, nu

            def get_physical(h: float, D: float, nu: float, *args) -> tuple[float, float]:
                E = 12 * D * (1 - nu ** 2) / h ** 3
                G = E / (2 * (1 + nu))
                return E, G

            self.phys_to_D = get_Ds
            self.D_to_phys = get_physical

            def get_params(h: float):
                return (*self.phys_to_D(h, self.E, self.G), self.beta)

            self.get_params = get_params

        elif self.atype == 'orthotropic':
            self.transform = orthotropic

            def get_Ds(h: float, E1: float, E2: float, G12: float,
                       nu12: float) -> tuple[float, float, float, float]:
                nu21 = E1 / E2 * nu12
                D11 = E1 * h ** 3 / (12 * (1 - nu12 * nu21))
                D66 = G12 * h ** 3 / 12
                return D11, nu12, E1 / E2, D66

            def get_physical(h: float, D11: float, nu12: float, E_rat: float,
                             D66: float, *args) -> tuple[float, float, float, float]:
                nu21 = E_rat * nu12
                E1 = D11 * 12 * (1 - nu12 * nu21) / h ** 3
                return E1, E1 / E_rat, nu12, D66 * 12.0 / h ** 3

            self.phys_to_D = get_Ds
            self.D_to_phys = get_physical

            def get_params(h: float):
                return (*self.phys_to_D(h, self.E1, self.E2, self.G12, self.nu12),
                        self.beta)

            self.get_params = get_params

        elif self.atype == 'orthotropic_d4':
            self.transform = orthotropic_d4

            def get_Ds(h: float, E1: float, E2: float, G12: float,
                       nu12: float, b1: float, b2: float,
                       b3: float, b4: float) -> tuple:
                E1 *= (1 + 1j * b1)
                E2 *= (1 + 1j * b2)
                G12 *= (1 + 1j * b3)
                nu12 *= (1 + 1j * b4)

                nu21 = E1 / E2 * nu12
                D11 = E1 * h ** 3 / (12 * (1 - nu12 * nu21))
                D66 = G12 * h ** 3 / 12

                Ds = np.array([D11, nu12, E1 / E2, D66])
                re_Ds = np.real(Ds)
                new_betas = np.tan(np.angle(Ds))
                return (*re_Ds, *new_betas)

            def get_physical(h: float, D11: float, nu12: float, E_rat: float,
                             D66: float, b1: float, b2: float,
                             b3: float, b4: float) -> tuple:
                D11 *= (1 + 1j * b1)
                nu12 *= (1 + 1j * b2)
                E_rat *= (1 + 1j * b3)
                D66 *= (1 + 1j * b4)

                nu21 = E_rat * nu12
                E1 = D11 * 12 * (1 - nu12 * nu21) / h ** 3
                phys = np.array([E1, E1 / E_rat, nu12, D66 * 12.0 / h ** 3])
                re_phys = np.real(phys)
                new_betas = np.tan(np.angle(phys))
                return (*re_phys, *new_betas)

            self.phys_to_D = get_Ds
            self.D_to_phys = get_physical

            def get_params(h: float):
                return self.phys_to_D(h, self.E1, self.E2, self.G12, self.nu12,
                                      self.b1, self.b2, self.b3, self.b4)

            self.get_params = get_params

    @staticmethod
    def create_material(params: MaterialParams, material_name: str) -> None:
        """
        Method to create a .json file for a material with given parameters and
        name in `materials` folder.

        Parameters
        ----------
        params : MaterialParams
            Parameters to be saved.
        material_name : str
            Name of the material to be saved.

        Returns
        -------
        None

        """
        mat = Material(params)
        materials_folder = os.path.join(get_jax_plate_dir(), 'materials')

        if not os.path.exists(materials_folder):
            os.mkdir(materials_folder)

        if not params.atype in ATYPES:
            raise ValueError(f'Invalid anisotropy type for material '
                             f'{material_name}. Supported options are: {ATYPES}.')

        fpath = os.path.join(materials_folder, material_name + '.json')

        with open(fpath, 'w') as file:
            json.dump(mat.__dict__, file, indent=4)

        return

    def __str__(self):
        return f'Material with {self.__dict__}.'
