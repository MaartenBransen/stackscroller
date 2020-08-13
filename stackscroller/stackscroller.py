# -*- coding: utf-8 -*-
"""
created by:     Maarten Bransen
email:          m.bransen@uu.nl
last updated:   13-08-2020
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np

class stackscroller:
    """
    scroll through 3D stack with highlighted features. Class instance must be
    stored to a global variable or the keybindings are lost.
    
    Parameters
    ----------
    stack : numpy.ndarray
        the pixel values, must have shape ([t,]z,y,x)
    features : pandas DataFrame, optional
        particle positions as formatted by from trackpy.locate for xyz or 
        from trackpy.link for xyzt. The default is no particles.
    pixel_aspect : tuple of float, optional
        (z,y,x) pixel size(ratio) for correct aspect ratio.  The default is
        (1,1,1).
    diameter : tuple of float, optional
        (z,y,x) diameters for feature highlighting. The default is
        (10,10,10).
    colormap : str, optional
        matplotlib colormap name for visualising the data. The default is
        'inferno'.
    colormap_percentile : tuple of 2 values from 0 to 100
        lower and upper percentile of the data values to use for the min and 
        max value limits of the colormap scaling
    timesteps : list of floats/ints
        time values for each frame to display. The default is the frame index 
        numbers.        
    print_options : bool
        prints the keybindings and instructions how to use to the terminal. The
        default is True.

    Returns
    -------
    stackscroller class instance
        dynamic stackscroller instance that can be called with keybindings
        to update the displayed frame
    """
    def __init__(
            self,
            stack,
            features = None,
            pixel_aspect = (1,1,1),
            diameter = (10,10,10),
            colormap = 'inferno',
            colormap_percentile = (0.01,99.99),
            timesteps = None,
            print_options = True
            ):
        """
        initialize the class instance, create the figure and link keybindings
        """
        #check and correct dimensionality
        if len(np.shape(stack)) == 3:
            self.stack = stack.reshape(1,np.shape(stack)[0],np.shape(stack)[1],np.shape(stack)[2])
        elif len(np.shape(stack)) == 4:
            self.stack = np.array(stack)
        else:
            raise ValueError('[stackscroller]: stack must be 3 or 4 dimensional')

        #check timesteps
        if timesteps == None:
            self.usetimesteps = False
        elif len(timesteps) != len(stack):
            raise ValueError('[Stackscroller]: number of timesteps must be equal to length of the stack.')
        else:
            self.usetimesteps = True
            self.timesteps = timesteps

        #store input
        self.shape = np.shape(self.stack)
        self.diameter = diameter
        self.pixel_aspect = pixel_aspect
        self.cmap = colormap
        
        #check if features are given, else create dummy dataframe
        if type(features) != pd.core.frame.DataFrame:
            self.features = pd.DataFrame(columns=['x','y','z','frame'])
        else:
            self.features = features.copy()
        if 'frame' not in self.features.columns:
            self.features['frame'] = [0]*len(features)
        
        #set starting positions to 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.t = 0
        
        #set defaults for changeable options
        self.axis = 0
        self.lines = True
        
        #create the figure window
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        
        #interactive handles
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        #color scaling
        self.norm = Normalize(vmin=np.percentile(stack,colormap_percentile[0]),vmax=np.percentile(stack,colormap_percentile[1]))
        
        #start
        if print_options:
            self.print_options()
        self.set_view_xy()
        self.update()
    
    def on_key(self,event):
        """
        when a key is pressed, increment and update plot
        """
        #set depth
        if event.key == 'up':
            self.slice = (self.slice + 1) % self.shape[self.axis]
        elif event.key == 'down':
            self.slice = (self.slice - 1) % self.shape[self.axis]
        elif event.key == 'ctrl+up':
            self.slice = (self.slice + 10) % self.shape[self.axis]
        elif event.key == 'ctrl+down':
            self.slice = (self.slice - 10) % self.shape[self.axis]

        #set time
        elif event.key == 'right':
            self.t = (self.t + 1) % self.shape[0]
            self.set_time()
        elif event.key == 'left':
            self.t = (self.t - 1) % self.shape[0]
            self.set_time()
        elif event.key == 'ctrl+right':
            self.t = (self.t + 10) % self.shape[0]
            self.set_time()
        elif event.key == 'ctrl+left':
            self.t = (self.t - 10) % self.shape[0]
            self.set_time()
        
        #set view
        elif event.key == '1':
            self.set_view_xy()
        elif event.key == '2':
            self.set_view_xz()
        elif event.key == '3':
            self.set_view_yz()
            
        #other
        elif event.key == 'h':
            self.print_options()
        
        elif event.key == 'l':
            self.lines = not self.lines
            if self.axis == 1:
                self.set_view_xy()
            elif self.axis == 2:
                self.set_view_xz()
            elif self.axis == 3:
                self.set_view_yz()
            
        #update the plot
        self.update()
    
    def print_options(self):
        """prints list of options"""
        print(' ---------------------------------------- ')
        print('|                 HOTKEYS                |')
        print('|                                        |')
        print('| up/down: slice through stack           |')
        print('| left/right: move through time          |')
        print('| ctrl + arrow keys: move 10 steps       |')
        print('| 1: display xy plane                    |')
        print('| 2: display xz plane                    |')
        print('| 3: display zy plane                    |')
        print('| l: toggle x/y/x lines                  |')
        print('| f: toggle fullscreen                   |')
        print('| h: print help/hotkeys                  |')
        print(' ---------------------------------------- ')
    
    def set_time(self):
        """set data to correct timestep"""
        self.data = self.oriented_stack[self.t]
        self.f = self.oriented_features[self.oriented_features[:,0]==self.t]
    
    def set_view_xy(self):
        """
        redraw the figure in the xy plane
        """
        #store slice to correct attribute
        if self.axis == 1:
            self.z = self.slice        
        elif self.axis == 2:
            self.y = self.slice
        elif self.axis == 3:
            self.x = self.slice
        
        #set and orient the data
        self.axis = 1
        self.oriented_stack = self.stack.copy()
        self.oriented_features = np.array([
                self.features['frame'],
                self.features['z'],
                self.features['y'],
                self.features['x']]).transpose()
        self.d = (self.diameter[0]*0.7,self.diameter[1],self.diameter[2])
        self.slice = self.z
        self.set_time()
        
        #reset the axes layout
        self.ax.clear()
        self.im = self.ax.imshow(
                self.data[self.slice,:,:],
                aspect = self.pixel_aspect[1]/self.pixel_aspect[2],
                norm = self.norm,
                cmap = self.cmap
                )
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        #plot lines for perpendicular view positions and title
        if self.lines:
            self.hline = self.ax.axhline(self.y,linestyle=':',color='white')
            self.vline = self.ax.axvline(self.x,linestyle=':',color='white')

        #set titleformat
        if self.usetimesteps:
            self.title = 'z position {} of {}, time {:.3g} of {:.3g} s'
        elif self.shape[0] > 1:
            self.title = 'z position {} of {}, frame {} of {}'
        else:
            self.title = 'z position {} of {}'

    def set_view_xz(self):
        """
        redraw the figure in the xz plane
        """
        #store slice to correct attribute
        if self.axis == 1:
            self.z = self.slice
        elif self.axis == 2:
            self.y = self.slice
        elif self.axis == 3:
            self.x = self.slice
        
        #set and orient the data
        self.axis = 2
        self.oriented_stack = np.swapaxes(self.stack,1,2)
        self.oriented_features = np.array([
                self.features['frame'],
                self.features['y'],
                self.features['z'],
                self.features['x']]).transpose()
        self.d = (self.diameter[1]*0.7,self.diameter[0],self.diameter[2])
        self.slice = self.y
        self.set_time()
        
        #reset the axes layout
        self.ax.clear()
        self.im = self.ax.imshow(
                self.data[self.slice,:,:],
                aspect = self.pixel_aspect[0]/self.pixel_aspect[2],
                norm = self.norm,
                cmap = self.cmap
                )
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('z')
        
        #plot lines for perpendicular view positions
        if self.lines:
            self.hline = self.ax.axhline(self.z,linestyle=':',color='white')
            self.vline = self.ax.axvline(self.x,linestyle=':',color='white')

        #set titleformat
        if self.usetimesteps:
            self.title = 'y position {} of {}, time {:.3g} of {:.3g} s'
        elif self.shape[0] > 1:
            self.title = 'y position {} of {}, frame {} of {}'
        else:
            self.title = 'y position {} of {}'
        
    def set_view_yz(self):
        """
        redraw the figure in the zy plane
        """
        #store slice to correct attribute
        if self.axis == 1:
            self.z = self.slice
        elif self.axis == 2:
            self.y = self.slice
        elif self.axis == 3:
            self.x = self.slice
        
        #set and orient the data
        self.axis = 3
        self.oriented_stack = np.swapaxes(self.stack.copy(),1,3)
        self.oriented_features = np.array([
                self.features['frame'],
                self.features['x'],
                self.features['y'],
                self.features['z']]).transpose()
        self.d = (self.diameter[2]*0.7,self.diameter[1],self.diameter[0])
        self.slice = self.x
        self.set_time()
        
        #reset the axes layout
        self.ax.clear()
        self.im = self.ax.imshow(
                self.data[self.slice,:,:],
                aspect = self.pixel_aspect[1]/self.pixel_aspect[0],
                norm = self.norm,
                cmap = self.cmap
                )
        self.ax.set_xlabel('z')
        self.ax.set_ylabel('y')
        
        #plot lines for perpendicular view positions
        if self.lines:
            self.hline = self.ax.axhline(self.y,linestyle=':',color='white')
            self.vline = self.ax.axvline(self.z,linestyle=':',color='white')

        #set titleformat
        if self.usetimesteps:
            self.title = 'x position {} of {}, time {:.3g} of {:.3g} s'
        elif self.shape[0] > 1:
            self.title = 'x position {} of {}, frame {} of {}'
        else:
            self.title = 'x position {} of {}'
    
    def update(self):
        """replot the figure and features"""
        #remove old patches if exist (in reverse order, this is necessary!)
        if len(self.ax.patches)!=0:
            [p.remove() for p in reversed(self.ax.patches)]
            
        #reset data
        self.im.set_data(self.data[self.slice,:,:])
        
        slicefeatures = self.f[np.logical_and(
                self.f[:,1] >= self.slice - self.d[0],
                self.f[:,1] <  self.slice + self.d[0])]
        
        #print features
        d = self.d[0]**4
        for x,y,z in zip(slicefeatures[:,3],slicefeatures[:,2],slicefeatures[:,1]):
            r = (1-(z-self.slice)**4/d)
            point = Ellipse((x,y),self.d[2]*r,self.d[1]*r,ec='r',fc='none')
            self.ax.add_patch(point)

        #title
        if self.usetimesteps:
            self.ax.set_title(self.title.format(self.slice,self.shape[self.axis],self.timesteps[self.t],self.timesteps[-1]))
        else:
            self.ax.set_title(self.title.format(self.slice,self.shape[self.axis],self.t+1,self.shape[0]))

        #draw figure
        self.im.axes.figure.canvas.draw()
                

class videoscroller:
    """
    scroll through xyt series with highlighted features. Class instance must be
    stored to a global variable or the keybindings are lost.
    
    Parameters
    ----------
    stack : numpy.ndarray
        the pixel values, must have shape (t,y,x)
    features : pandas DataFrame, optional
        particle positions as formatted by from trackpy.locate/batch. The 
        default is no particles.
    pixel_aspect : tuple of float, optional
        (y,x) pixel size(ratio) for correct aspect ratio.  The default is
        (1,1).
    diameter : tuple of float, optional
        (y,x) diameters for feature highlighting. The default is
        (10,10).
    colormap : str, optional
        matplotlib colormap name for visualising the data. The default is
        'inferno'.
    colormap_percentile : tuple of 2 values from 0 to 100
        lower and upper percentile of the data values to use for the min and 
        max value limits of the colormap scaling
    timesteps : list of floats/ints
        list of time stamps for the video frames to display. The default is the
        frame index numbers.
    print_options : bool
        prints the keybindings and instructions how to use to the terminal. The
        default is True.

    Returns
    -------
    stackscroller class instance
        dynamic stackscroller instance that can be called with keybindings
        to update the displayed frame
    """
    def __init__(
            self,
            stack,
            features = None,
            pixel_aspect = (1,1),
            diameter = (10,10),
            colormap = 'inferno',
            colormap_percentile = (0.01,99.99),
            timesteps = None,
            print_options = True
            ):
        """
        initialize the class instance, create the figure and link keybindings
        """

        #store input
        self.stack = np.array(stack)
        self.shape = np.shape(self.stack)
        self.diameter = diameter
        self.pixel_aspect = pixel_aspect
        self.cmap = colormap

                #check timesteps
        if type(timesteps) == type(None):
            self.usetimesteps = False
        elif len(timesteps) != len(stack):
            raise ValueError('[Stackscroller]: number of timesteps must be equal to length of the stack.')
        else:
            self.usetimesteps = True
            self.timesteps = timesteps

        #check if features are given, else create dummy dataframe
        if type(features) != pd.core.frame.DataFrame:
            self.features = pd.DataFrame(columns=['x','y','frame'])
        else:
            self.features = features.copy()
        if 'frame' not in self.features.columns:
            self.features['frame'] = [0]*len(features)
        
        #set starting positions to 0
        self.x = 0
        self.y = 0
        self.t = 0

        #color scaling
        self.norm = Normalize(vmin=np.percentile(stack,colormap_percentile[0]),vmax=np.percentile(stack,colormap_percentile[1]))

        #create the figure window
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(
                self.stack[self.t],
                aspect = self.pixel_aspect[0]/self.pixel_aspect[1],
                norm = self.norm,
                cmap = self.cmap
                )
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        #interactive handles
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


        #start
        if print_options:
            self.print_options()
        self.update()
    
    def on_key(self,event):
        """
        when a key is pressed, increment and update plot
        """
        #set time
        if event.key == 'right':
            self.t = (self.t + 1) % self.shape[0]
        elif event.key == 'left':
            self.t = (self.t - 1) % self.shape[0]
        elif event.key == 'ctrl+right':
            self.t = (self.t + 10) % self.shape[0]
        elif event.key == 'ctrl+left':
            self.t = (self.t - 10) % self.shape[0]

        #other
        elif event.key == 'h':
            self.print_options()

        #update the plot
        self.update()
    
    def print_options(self):
        """prints list of options"""
        print(' ---------------------------------------- ')
        print('|                 HOTKEYS                |')
        print('|                                        |')
        print('| left/right: move through time          |')
        print('| ctrl + arrow keys: move 10 steps       |')
        print('| f: toggle fullscreen                   |')
        print('| h: print help/hotkeys                  |')
        print(' ---------------------------------------- ')

    def update(self):
        """replot the figure and features"""
        #remove old patches if exist (in reverse order, this is necessary!)
        if len(self.ax.patches)!=0:
            [p.remove() for p in reversed(self.ax.patches)]
            
        #reset data
        self.im.set_data(self.stack[self.t])

        framefeatures = self.features[self.features['frame']==self.t]

        #print features
        for x,y in zip(framefeatures['x'],framefeatures['y']):
            point = Ellipse((x,y),self.diameter[1],self.diameter[0],ec='r',fc='none')
            self.ax.add_patch(point)

        #title
        if self.usetimesteps:
            self.ax.set_title('time: {:.3f} of {:.3f} s'.format(self.timesteps[self.t],self.timesteps[-1]))
        else:
            self.ax.set_title('frame {} of {}'.format(self.t,self.shape[0]))

        #draw figure

        self.im.axes.figure.canvas.draw()
       


class videoscroller_old(object):
    """
    scroll through 2D timeseries with highlighted features
    
    @parameters:
        figure,axes:    plt.subplots handles
        images:         ndarray of (t,y,x)
        features:       pandas dataframe from trackpy.locate or None
        timestamps:     list of floats, timestamps for the frames
        pixel_aspect:         aspect ratio of pixels
        diameter:    diameters for feature highlighting
        normalize:      tuple, lower and upper percentile of data to scale
                        colormap to
    
    @dependencies:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.colors import Normalize
        import scipy as sp
        import pandas
        
    @returns:
        dynamic videoscroller instance that can be called with keybindings to
        update the displayed frame
        
    """
    
    def __init__(self,figure,axes,images,features=pd.DataFrame(columns=['x','y','frame']),
                 timestamps=[],aspect=1,xdiam=3,ydiam=3,normalize=(0,100)):
        """initialize the stackscroller class, assign relevant variables"""
        
        #assign properties
        self.fig = figure
        self.ax = axes
        self.frame = 0       
        self.images = images
        self.features = features
        self.t = timestamps
        self.xdiam = xdiam
        self.ydiam = ydiam
        self.nframes = len(images)
        
        #create the imshow
        colorscale = Normalize(vmin=np.percentile(images,normalize[0]),vmax=np.percentile(images,normalize[1]))
        self.im = self.ax.imshow(self.images[self.frame],aspect=aspect,norm=colorscale)
        self.update()
        
    def onscroll(self, event):
        """when scroll wheel is used, scroll through time"""
        if event.button == 'up':
            self.frame = (self.frame + 1) % self.nframes
        else:
            self.frame = (self.frame - 1) % self.nframes
        self.update()
    
    def onarrow(self, event):
        """
        when arrow up or down button are pressed, skip 10 frames, when arrow
        left or right are used scroll through time
        """
        if event.key == 'up':
            self.frame = (self.frame + 10) % self.nframes
        elif event.key == 'down':
            self.frame = (self.frame - 10) % self.nframes
        elif event.key == 'right':
            self.frame = (self.frame + 1) % self.nframes
        elif event.key == 'left':
            self.frame = (self.frame - 1) % self.nframes
        self.update()

    def update(self):
        """replot the figure and features"""
        #remove old patches if exist (in reverse order, this is necessary!)
        if len(self.ax.patches)!=0:
            [p.remove() for p in reversed(self.ax.patches)]
            
        #reset data and features
        self.im.set_data(self.images[self.frame])
        framefeatures = self.features.loc[self.features['frame']==self.frame]
        for x,y in zip(framefeatures['x'],framefeatures['y']):
            point = Ellipse((x,y),self.xdiam,self.ydiam,ec='r',fc='none')
            self.ax.add_patch(point)
        
        #put current time in image title, use frameindex when no timestamps are given
        if len(self.t) != self.nframes:
            self.ax.set_title('t: {0} of {1}'.format(self.frame,self.nframes))
        else:
            self.ax.set_title('t: {:0.2f} of {:0.2f}'.format(self.t[self.frame],self.t[-1]))
        
        self.im.axes.figure.canvas.draw()
