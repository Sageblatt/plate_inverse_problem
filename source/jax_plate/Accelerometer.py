from dataclasses import dataclass
import json
import os

from jax_plate.Utils import get_jax_plate_dir

@dataclass
class AccelerometerParams:
    """
    Class that represents the list of parameters for given accelerometer type.
    """
    mass: float
    radius: float
    height: float = None


class Accelerometer:
    """
    Class that represents accelerometer and it's properties, provides interface
    to *.json files.
    """
    def __init__(self, name_or_params: str | AccelerometerParams):
        """
        Constructor method.

        Parameters
        ----------
        name_or_params : str | AccelerometerParams
            Name of the accelerometer to search for in `accelerometers` folder
            or a dataclass with corresponding values.

        Returns
        -------
        None

        """
        params = None

        if isinstance(name_or_params, str):
            fpath = os.path.join(get_jax_plate_dir(), 'accelerometers',
                                 name_or_params + '.json')

            if os.path.exists(fpath):
                with open(fpath, 'r') as file:
                    params = json.load(file)

            else:
                raise ValueError(f'Could not find file {name_or_params}.json '
                                 'in `accelerometers` folder.')

        elif isinstance(name_or_params, AccelerometerParams):
            params = name_or_params.__dict__

        else:
            raise TypeError('Argument `name_or_params` should have type '
                            '`str` or `AccelerometerParams.`')

        self.mass = params['mass']
        self.radius = params['radius']
        self.height = params['height']


    @staticmethod
    def create_accelerometer(params: AccelerometerParams, accelerometer_name: str) -> None:
        """
        Method to create a .json file for a accelerometer with given parameters
        and name in `accelerometers` folder.

        Parameters
        ----------
        params : AccelerometerParams
            Parameters to be saved.
        accelerometer_name : str
            Name of the accelerometer to be saved.

        Returns
        -------
        None

        """
        accel_folder = os.path.join(get_jax_plate_dir(), 'accelerometers')

        if not os.path.exists(accel_folder):
            os.mkdir(accel_folder)

        fpath = os.path.join(accel_folder, accelerometer_name + '.json')

        with open(fpath, 'w') as file:
            json.dump(params.__dict__, file, indent=4)

        return

    def __str__(self):
        return f'Accelerometer with {self.__dict__}.'
