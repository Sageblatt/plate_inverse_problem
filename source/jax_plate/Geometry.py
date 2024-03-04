from dataclasses import dataclass
import os
import re
import shutil
from jax_plate.Utils import get_jax_plate_dir
from jax_plate.Accelerometer import Accelerometer, AccelerometerParams


# Available options are documented in Geometry.__init__ docstring.
TEMPLATES = ['sh_r', 'sh_i', 'symm']


@dataclass
class GeometryParams:
    """Class that represents parameters of a simple rectangular plate."""
    length: float
    width: float
    height: float
    accel_x: float = None # TODO: these can be the coordinates of a test point without accel.
    accel_y: float = None # None if the test point lies on a symmetry line.


class Geometry:
    """
    Class that stores plate's geometrical properties and provides interface to
    work with .edp files from Python code.
    """
    def __init__(self,
                 edp_or_template: str | os.PathLike,
                 accelerometer: Accelerometer = None,
                 params: GeometryParams = None,
                 *,
                 height: float = None,
                 export_vtk: bool = False):
        """
        Constructor method.

        Allows to create Geometry objects from edp files, updating class
        attributes automatically from chosen script or to create them using
        templates with specified parameters and accelerometer.
        Available templates:
            1) 'sh_r' (shifted_real) -- accelerometer is placed at a custom
                position (requires both accel_x and accel_y to be specified).
            2) 'sh_i' (shifted_ideal) -- accelerometer is placed in a corner
                of a plate.
            3) 'symm' (symmetrical) -- accelerometer is placed at a symmeterical
                position with respect to width (requires only accel_x to be
                specified).

        Parameters
        ----------
        edp_or_template : str | os.PathLike
            Name of a template from a list above or a path to .edp scipt with
            geometry.
        accelerometer : Accelerometer, optional
            Accelerometer object, which contains radius of the accelerometer
            (required when using a template). The default is None.
        params : GeometryParams, optional
            GeometryParams object, which contains plate's main
            geometrical properties (required when using a template).
            The default is None.
        height : float, optional
            Height of a plate, required when params is `None` and creating from
            script. Ignored when params is not `None`. The default is None.
        export_vtk : bool, optional
            Flag, enables saving resulting geometry to
            JAX_PLATE_DIR/geometry/export.vtu file. The default is False.

        Returns
        -------
        None

        """
        if edp_or_template in TEMPLATES:
            chosen_template = os.path.join(get_jax_plate_dir(), 'geometry',
                                           f'{edp_or_template}.edp')
            self.current_file = Geometry._create_temp_file(chosen_template)

            if params is None:
                raise ValueError('`params` argument cannot be None when '
                                 'using a template.')

            elif accelerometer is None:
                raise ValueError('`accelerometer` argument cannot be None when '
                                 'using a template.')

            if edp_or_template == TEMPLATES[0]:
                if None in (params.accel_x, params.accel_y):
                    raise ValueError('Both coordinates of accelerometer '
                                     'should be specified for the template '
                                     f'{TEMPLATES[0]}.')
                params.accel_x = params.accel_x
                params.accel_y = params.width/2 - params.accel_y

            elif edp_or_template == TEMPLATES[1]:
                if params.accel_y is not None or params.accel_x is not None:
                    raise ValueError('Both coordinates of accelerometer '
                                     'should be None for the template '
                                     f'{TEMPLATES[1]}.')
                params.accel_x = accelerometer.radius
                params.accel_y = params.width/2 - accelerometer.radius

            elif edp_or_template == TEMPLATES[2]:
                if params.accel_y is not None:
                    raise ValueError('`y` coordinate of the accelerometer'
                                     'should be None for the template'
                                     f'{TEMPLATES[2]}.')
                elif params.accel_x is None:
                    raise ValueError('`x` coordinate of the accelerometer'
                                     'should not be None for the template'
                                     f'{TEMPLATES[2]}.')
                params.accel_y = 0

        elif os.path.exists(edp_or_template):
            self.current_file = Geometry._create_temp_file(edp_or_template)

            if params is None and height is None:
                raise ValueError('Height of the plate should be specified via'
                                 '`params` arg or `height` kwarg when loading'
                                 'from a file.')

            if params is None:
                params = GeometryParams(None, None, height, None, None)

            if accelerometer is None:
                accelerometer = Accelerometer(AccelerometerParams(None, None, None))

        else:
            if os.path.splitext(edp_or_template)[1] == '.edp':
                raise FileNotFoundError(f'Could not find file {edp_or_template}.')
            else:
                raise ValueError(f'Could not find template {edp_or_template}.'
                                 f'Valid options are: {TEMPLATES}. For '
                                 'description see the docstring for '
                                 'Geometry.__init__.')

        self.length = params.length
        self.width = params.width
        self.height = params.height
        self.accel_x = params.accel_x
        self.accel_y = params.accel_y
        self.accel_r = accelerometer.radius

        self._generate_edp(export_vtk)
        return

    @staticmethod
    def _create_temp_file(path_to_original: str | os.PathLike) -> str:
        """
        Copies .edp file into geometry folder for further manipulations.

        Parameters
        ----------
        path_to_original : str | os.PathLike
            Path to the .edp file to be copied.

        Raises
        ------
        ValueError
            If the path_to_original does not lead to .edp file.

        Returns
        -------
        str
            Path of the created _temp.edp file.

        """
        if os.path.splitext(path_to_original)[1] != '.edp':
            raise ValueError('Temp file can be created only from a valid .edp'
                             'file.')

        dst = os.path.join(get_jax_plate_dir(), 'geometry', '_temp.edp')
        shutil.copy(path_to_original, dst)
        return dst

    def _generate_edp(self, export_vtk: bool = False) -> None:
        """
        Changes parameters in self.current_file file and get required values
        from it.

        Parameters
        ----------
        export_vtk : bool, optional
            See Geometry.__init__ method description. The default is False.

        Returns
        -------
        None

        """
        kwords = {'length': 'Lx', 'width': 'Ly', 'accel_r': 'rAccel',
                  'accel_x': 'offsetAccelX', 'accel_y': 'offsetAccelY'}

        with open(self.current_file, 'r+') as edp:
            data = edp.read()

            for key in kwords:
                attr = getattr(self, key)
                if attr is not None:
                    (data, count) = re.subn(f'.*real.*{kwords[key]}.*=.*;',
                                            f'real {kwords[key]} = {attr:.5e};',
                                            data, count=1)
                    if count == 0:
                        raise RuntimeError(f'Could not replace {kwords[key]} '
                                           f'parameter in .edp file '
                                           f'{self.current_file} with given '
                                           f'value {attr:.5e} as it was not '
                                           'found in the original script.')

                else:
                    match = re.search('.*real.*{kwords[key].*=.*;}', data)
                    if match is None:
                        # raise RuntimeError(f'Could not find {kwords[key]}'
                        #                    'parameter in .edp file '
                        #                    f'{self.current_file}, loading the '
                        #                    'value to the solver cannot be '
                        #                    'completed.')
                        continue

                    attr_str = match.group()
                    value = float(attr_str.split('=')[1].strip(';'))
                    setattr(self, key, value)

            if export_vtk:
                (data, count) = re.subn(r'//savevtk\(".*"',
                                        'savevtk("export.vtu"',
                                        data, count=1)
                if count == 0:
                    raise RuntimeError('Could not export geometry to .vtu '
                                       'file as a line starting with '
                                       '`//savevtk` was not found in the '
                                       'original script.')
            # print(data)
            edp.seek(0)
            edp.write(data)
            edp.truncate()

        return

    def __str__(self):
        return f'Geometry with {self.__dict__}.'
