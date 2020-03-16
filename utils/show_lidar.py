import sys
sys.path.append("..")
import numpy as np
import mayavi.mlab
from utils import data_provider


# project pointcloud to 3D plane
def show_pc(pointcloud: list,
             _type: str     # 'reflectence', 'distance' or 'depth'
             ):
    '''
    parameters:
        pointcloud: data to project
        type: which value to be project
    '''
    assert _type in {"depth", "height", "reflectance"}, \
                    'val must be one of {"depth", "height", "reflectance"}'

    x = pointcloud[0]
    y = pointcloud[1]
    z = pointcloud[2]
    
    if _type=='reflectance':
        value = pointcloud[3]
    if _type=='distance':
        value = pointcloud[4]
    if _type=='depth':
        value = pointcloud[5]

    '''
    # normalize the value in color range
    v_min = value.min()
    v_range = value.max() - v_min
    value = (value - v_min) / v_range
    '''

    # set figure, with background color (0,0,0) -> white, size -> (500,500)
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(500, 500))
    
    mayavi.mlab.points3d(x, y, z, value,          # Values used for Color
                        mode="point",
                        colormap='spectral', # bone, copper, gnuplot, spectral
                        #color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                        figure=fig
                        )
    
    x=np.linspace(-5,5,100)
    y=np.linspace(-5,5,100)
    z=np.linspace(-5,5,100)
    mayavi.mlab.plot3d(x,y,z)
    mayavi.mlab.show()


if __name__ == "__main__":
    filename = "../data/bin/um_000000.bin"
    print(filename,' for test: \n')
    data = data_provider.read_pc2array(filename) # , height=[-1.75,-1.55], font=True
    show_pc(data, "reflectance")
