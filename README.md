# stackscroller
Library for visualizing 2-dimensional and 3-dimensional time series with optionally including particle tracking results

## Info
- created by:     Maarten Bransen
- email:          m.bransen@uu.nl

## Installation instructions
Download the `stackscroller` folder and place it in your `site-packages` location of your Anaconda installation. If you are unsure where this is located you can find the path of any already installed package, e.g. using numpy:
```
import numpy
print(numpy.__file__)
```
which may print something like
```
'<current user>\\AppData\\Local\\Continuum\\anaconda3\\lib\\site-packages\\numpy\\__init__.py'
```

## Usage
There are two classes:
- stackscroller: for visualizing a 3-dimensional stack (or a time series of 3-dimensional stacks)
- videoscroller: for visualizing a 2-dimensional time series

When creating a class instance, it *must* be stored to a global variable, otherwise the python garbage collector comes and deletes all of our information needed to scroll through the data. Additionally, the Spyder ide must be set up to open figures in a separate window. 

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
