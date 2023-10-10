import os
import json
from dataclasses import dataclass
from .Utils import get_jax_plate_dir
from .ParamTransforms import isotropic, orthotropic

# TODO: add support for anisotropic materials

ATYPES = {'isotropic': ['E', 'G', 'beta'],
          'orthotropic': ['E1', 'E2', 'G12', 'nu12', 'beta']}

@dataclass
class MaterialParams:
    """Class that represents the list of parameters for given material."""
    density: float
    atype: str # Anosotropy type, possible options are listed in ATYPES dict

    E: float = None
    G: float = None

    E1: float = None
    E2: float = None
    G12: float = None
    nu12: float = None

    beta: float = None


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

            else:
                raise ValueError(f'Could not find file {name_or_params}.json '
                                 'in `materials` folder.')

        elif isinstance(name_or_params, MaterialParams):
            params = name_or_params.__dict__

        else:
            raise TypeError('Argument `name_or_params` should have type '
                            '`str` or `MaterialParams.`')

        if params['atype'] in ATYPES:
            self.density = params['density']
            self.atype = params['atype']
            for param in ATYPES[self.atype]:
                setattr(self, param, params[param])

            if self.atype == 'isotropic':
                self.transform = isotropic

            elif self.atype == 'orthotropic':
                self.transform = orthotropic

        else:
            raise ValueError(f'Invalid anisotropy type for material '
                             f'{name_or_params}. Supported options are: {ATYPES}.')
        return

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
