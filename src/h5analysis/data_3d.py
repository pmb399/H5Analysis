import numpy as np
from .ReadData import Data
from .util import check_key_in_dict
from .simplemath import apply_offset, grid_data2d
from .readutil import stack_norm

def load_3d(config, file, stack, arg, xoffset=None, xcoffset=None, yoffset=None, ycoffset=None,grid_x=[None, None, None],grid_y=[None, None,None],norm_by=None):
    """Internal function to load STACK data
    
        Parameters
        ----------
    """
    
    if not check_key_in_dict(stack,config.h5dict):
        raise Exception("Data Stream undefined.")
    else:
        if not config.h5dict[stack]['type'] == "STACK":
            raise Exception("Need to specify an image stack")
        else:

            # Place all loaded REIXS objects in data dictionary
            data = dict()

            data[arg] = Data(config,file,arg)
            data[arg].scan = arg
            all_data = data[arg].Scan(stack)

            # Apply offset
            x_data = apply_offset(all_data[f"{stack}_scale1"], xoffset, xcoffset)
            y_data = apply_offset(all_data[f"{stack}_scale2"], yoffset, ycoffset)

            # Normalize MCA data by SCA
            if not isinstance(norm_by,type(None)):
                norm_data = data[arg].Scan(norm_by)
                normalization = norm_data[norm_by]
                my_stack = stack_norm(all_data[stack],normalization)
            else:
                my_stack = all_data[stack]

            stack_grid = list()
            for i,img in enumerate(my_stack):
                xmin, xmax, ymin, ymax, new_x, new_y, new_z = grid_data2d(x_data, y_data, img, grid_x=grid_x,grid_y=grid_y)
                stack_grid.append(new_z)

            data[arg].stack = np.stack(tuple(stack_grid))
            data[arg].x_min = xmin
            data[arg].x_max = xmax
            data[arg].y_min = ymin
            data[arg].y_max = ymax

            return data