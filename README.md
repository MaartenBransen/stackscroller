# stackscroller
Library for visualizing 2-dimensional and 3-dimensional time series with optionally including particle tracking results

**[The documentation can be found here](https://maartenbransen.github.io/stackscroller/index.html)**

## Info
- created by:     Maarten Bransen
- email:          m.bransen@uu.nl

## Installation
### PIP
This package can be installed directly from GitHub using pip:
```
pip install git+https://github.com/MaartenBransen/stackscroller
```
### Anaconda
When using the Anaconda distribution, it is safer to run the conda version of pip as follows:
```
conda install pip
conda install git
pip install git+https://github.com/MaartenBransen/stackscroller
```
### Updating
Updating to the most recent version can be done by running pip with the `--upgrade`  flag:
```
pip install --upgrade git+https://github.com/MaartenBransen/stackscroller
```

## Usage
There are two classes:
- [stackscroller](https://maartenbransen.github.io/stackscroller/index.html#stackscroller.stackscroller): for visualizing a 3-dimensional stack (or a time series of 3-dimensional stacks)
- [videoscroller](https://maartenbransen.github.io/stackscroller/index.html#stackscroller.videoscroller): for visualizing a 2-dimensional time series

When creating a class instance, it *must* be stored to a global variable, otherwise the python garbage collector comes and deletes all of our information needed to scroll through the data. Additionally, when using the Spyder IDE, it must be set up to open figures in a separate window. 

Example usage:
```
from stackscroller import stackscroller
import numpy as np

#create example data with 10 time steps, and 50x128x512 voxels in z,y,x respectively
data = np.random.rand(10,50,128,512)

#create the scroller object with a z-pixel size 3 times that of the x and y pixel size
myscroller1 = stackscroller(data,pixel_aspect=(3,1,1))
```

you can now scroll through z and time using the arrow keys, and switch the viewing direction with 1, 2 or 3 for xy, xz and yz respectively.
