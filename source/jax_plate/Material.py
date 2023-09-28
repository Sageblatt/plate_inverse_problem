from dataclasses import dataclass
import json
import os
import ParamTransforms

# TODO: add support for anisotropic materials

ATYPES = {'isotropic'}

@dataclass
class MaterialParams:
    """
    Class that represents the list of parameters for given material.
    """
    density: float
    E: float
    G: float
    beta: float
    atype: str # Anosotropy type, can be only 'isotropic' for now
    

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
            fpath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 'materials', name_or_params + '.json')
            
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
            if params['atype'] == 'isotropic':
                self.transform = ParamTransforms.isotropic
                self.E = params['E']
                self.G = params['G']
                self.density = params['density']
                self.beta = params['beta']
                
        
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
        materials_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        'materials')
        
        if not os.path.exists(materials_folder):
            os.mkdir(materials_folder)
        
        if not params.atype in ATYPES:
            raise ValueError(f'Invalid anisotropy type for material '
                             f'{material_name}. Supported options are: {ATYPES}.')
        
        fpath = os.path.join(materials_folder, material_name + '.json')
        
        with open(fpath, 'w') as file:
            json.dump(params.__dict__, file, indent=4)
        
        return


if __name__ == '__main__':
    params = MaterialParams(100, 102, 1241, 0.01, 'isotropic')
    print(params)
    
    name = 'Example_material'
    Material.create_material(params, name)
    
    m = Material(name)
    print(m.E)
        