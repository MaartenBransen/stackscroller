# -*- coding: utf-8 -*-
"""
created by:     Maarten Bransen
email:          m.bransen@uu.nl
last updated:   13-08-2020
"""

import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
import numpy as np

class stackscroller:
    """
    scroll through 3D stack with highlighted features. Class instance must be
    stored to a global variable or the keybindings are lost.
    
    Parameters
    ----------
    stack : numpy.ndarray
        the pixel values, must be three- or four dimensional with dimension 
        order `([t,]z,y,x)`.
    features : pandas DataFrame, optional
        dataframe with feature positions to highlight with circles on the 
        image. Must contain 3 columns with feature positions (in units of 
        pixels) for the z, y and x dimensions, with column headers matching 
        `pos_cols`. Additionally, a column indicating to which (integer) time 
        index the features belong can be provided with a column header matching
        `frame_col`. This matches the output of trackpy.locate for xyz or 
        from trackpy.link for xyzt. The highlight color for features is `r` 
        (red) by default, but can be set on a per-feature basis by including a
        column named `stackscroller_color` in the DataFrame where each item is
        a valid color specifier for Matplotlib patch colors. The default is 
        not to highlight any features.
    pixel_aspect : tuple of float, optional
        (z,y,x) pixel size(ratio) for correct aspect ratio.  The default is
        `(1,1,1)`.
    diameter : tuple of float, optional
        (z,y,x) diameters for feature highlighting. The default is
        `(10,10,10)`.
    pos_cols : tuple of str, optional
        names/headers of the columns in `features` containing the positional 
        data (in units of pixels) in (z,y,x) order. When `features=None`, 
        `pos_cols` is ignored. The default is `('z','y','x')`.
    frame_col : str, optional
        name/header of the column in `features` containing the integer index
        along the time dimension in `stack` the features correspond to. When
        `features=None` or if `stack` has 3 dimensions, `frame_col` is ignored.
        The default is `frame`.
    colormap : str, optional
        matplotlib colormap name for visualising the data. The default is
        `'inferno'`.
    colormap_percentile : tuple of 2 values from 0 to 100
        lower and upper percentile of the data values to use for the min and 
        max value limits of the colormap scaling. The default is `(0.01,99.99)`
    timesteps : list of floats/ints
        time values for each frame to display. The default is the frame index 
        numbers.        
    print_options : bool
        prints the keybindings and instructions how to use to the terminal. The
        default is `True`.

    Returns
    -------
    stackscroller :
        dynamic stackscroller instance that can be called with keybindings
        to update the displayed frame
    """
    def __init__(
            self,
            stack,
            features = None,
            pixel_aspect = (1,1,1),
            diameter = (10,10,10),
            pos_cols = ('z','y','x'),
            frame_col = 'frame',
            colormap = 'inferno',
            colormap_percentile = (0.01,99.99),
            timesteps = None,
            print_options = True
            ):
        
        #check and correct dimensionality
        if len(np.shape(stack)) == 3:
            self.stack = stack.reshape(
                1,
                np.shape(stack)[0],
                np.shape(stack)[1],
                np.shape(stack)[2]
            )
        elif len(np.shape(stack)) == 4:
            self.stack = np.array(stack)
        else:
            raise ValueError('stack must be 3 or 4 dimensional')

        #check timesteps
        if type(timesteps) == type(None):
            self.use_timesteps = False
        elif len(timesteps) != len(stack):
            raise ValueError('[Stackscroller]: number of timesteps must be '+
                             'equal to length of the stack.')
        else:
            self.use_timesteps = True
            self.timesteps = timesteps

        #store input
        self.shape = np.shape(self.stack)
        self.diameter = diameter
        self.pixel_aspect = pixel_aspect
        self.cmap = colormap
        
        #check if features are given
        if features is None:
            self.use_features = False
        #if they are, store dataframe to attribute
        else:
            self.use_features = True
            self.cols = [frame_col]+list(pos_cols)
            
            #check if there's a frame column in case of no time dimension
            if frame_col not in features.columns:
                features = features.copy()
                features[frame_col] = [0]*len(features)
            
            self.features = features[self.cols].copy()
            
            #check for color column, if it exists split it into timesteps
            if 'stackscroller_color' in features.columns:
                self.colors = [
                    features.loc[
                        features[frame_col]==t
                    ]['stackscroller_color'].to_numpy() \
                        for t in range(self.shape[0])
                ]
            else:
                self.colors = False
                self.c = 'r'
        
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
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        #color scaling
        self.norm = Normalize(
            vmin=np.percentile(stack,colormap_percentile[0]),
            vmax=np.percentile(stack,colormap_percentile[1])
        )
        
        #start
        if print_options:
            self._print_options()
        self._set_view_xy()
        self._update()
        plt.show(block=False)
    
    def __repr__(self):
        """interpreter string representation of stackscroller object"""
        return f"<stackscroller.stackscroller()> with shape {self.shape}"
    
    def __del__(self):
        """close figure and give warning upon garbage collection"""
        plt.close(self.fig)
        #print('closing',self)
    
    def _on_key(self,event):
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
            self._set_time()
        elif event.key == 'left':
            self.t = (self.t - 1) % self.shape[0]
            self._set_time()
        elif event.key == 'ctrl+right':
            self.t = (self.t + 10) % self.shape[0]
            self._set_time()
        elif event.key == 'ctrl+left':
            self.t = (self.t - 10) % self.shape[0]
            self._set_time()
        
        #set view
        elif event.key == '1':
            self._set_view_xy()
        elif event.key == '2':
            self._set_view_xz()
        elif event.key == '3':
            self._set_view_yz()
            
        #other
        elif event.key == 'h':
            self._print_options()
        
        elif event.key == 'l':
            self.lines = not self.lines
            if self.axis == 1:
                self._set_view_xy()
            elif self.axis == 2:
                self._set_view_xz()
            elif self.axis == 3:
                self._set_view_yz()
            
        #update the plot
        self._update()
    
    def _print_options(self):
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
    
    def _set_time(self):
        """set data to correct timestep"""
        self.data = self.oriented_stack[self.t]
        if self.use_features:
            self.f = self.oriented_features[self.t]
    
    def _set_view_xy(self):
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
        if self.use_features:
            #set features to list (t) of numpy.arrays (coords) with pos cols
            #switched to the correct order for the current view
            self.oriented_features = \
                [
                    self.features.loc[
                        self.features[self.cols[0]]==t
                        ][self.cols[1:]].to_numpy() \
                    for t in range(self.shape[0])
                ]
            self.d = (self.diameter[0]*0.7,self.diameter[1],self.diameter[2])
        self.slice = self.z
        self._set_time()
        
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
        if self.use_timesteps:
            self.title = 'z position {} of {}, time {:.3g} of {:.3g} s'
        elif self.shape[0] > 1:
            self.title = 'z position {} of {}, frame {} of {}'
        else:
            self.title = 'z position {} of {}'

    def _set_view_xz(self):
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
        if self.use_features:
            #set features to list (t) of numpy.arrays (coords) with pos cols
            #switched to the correct order for the current view
            self.oriented_features = \
                [
                    self.features.loc[
                        self.features[self.cols[0]]==t
                        ][[self.cols[2],self.cols[1],self.cols[3]]].to_numpy()\
                    for t in range(self.shape[0])
                ]
            self.d = (self.diameter[1]*0.7,self.diameter[0],self.diameter[2])
        self.slice = self.y
        self._set_time()
        
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
        if self.use_timesteps:
            self.title = 'y position {} of {}, time {:.3g} of {:.3g} s'
        elif self.shape[0] > 1:
            self.title = 'y position {} of {}, frame {} of {}'
        else:
            self.title = 'y position {} of {}'
        
    def _set_view_yz(self):
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
        if self.use_features:
            #set features to list (t) of numpy.arrays (coords) with pos cols
            #switched to the correct order for the current view
            self.oriented_features = \
                [
                    self.features.loc[
                        self.features[self.cols[0]]==t
                        ][[self.cols[3],self.cols[2],self.cols[1]]].to_numpy()\
                    for t in range(self.shape[0])
                ]
            self.d = (self.diameter[2]*0.7,self.diameter[1],self.diameter[0])
        self.slice = self.x
        self._set_time()
        
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
        if self.use_timesteps:
            self.title = 'x position {} of {}, time {:.3g} of {:.3g} s'
        elif self.shape[0] > 1:
            self.title = 'x position {} of {}, frame {} of {}'
        else:
            self.title = 'x position {} of {}'
    
    def _update(self):
        """replot the figure and features"""   
        #reset data
        self.im.set_data(self.data[self.slice,:,:])
        
        #add features
        if self.use_features:
            
            #remove EllipseCollection if it exists
            try:
                self.ec.remove()
            except:
                AttributeError
            
            #select features to display
            mask = np.logical_and(
                self.f[:,0] >= self.slice - self.d[0],
                self.f[:,0] <  self.slice + self.d[0]
            )
            slicefeatures = self.f[mask]
            
            if self.colors:
                self.c = self.colors[self.t][mask]
            
            #calculate shrink factor for xy diameter, depending on particle 
            #z offset from current display slice
            r = (1-(slicefeatures[:,0]-self.slice)**4/self.d[0]**4)
            
            #plot features for current frame
            self.ec = EllipseCollection(
                r*self.d[2],
                r*self.d[1],
                0,
                units='xy',
                offsets = slicefeatures[:,[2,1]],
                transOffset=self.ax.transData,
                edgecolors=self.c,
                facecolors='none'
            )

            self.ax.add_collection(self.ec)

        #title
        if self.use_timesteps:
            self.ax.set_title(
                self.title.format(
                    self.slice,
                    self.shape[self.axis],
                    self.timesteps[self.t],
                    self.timesteps[-1]
                )
            )
        else:
            self.ax.set_title(
                self.title.format(
                    self.slice,
                    self.shape[self.axis],
                    self.t+1,
                    self.shape[0]
                )
            )

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
        dataframe with feature positions to highlight with circles on the 
        image. Must contain 2 columns with feature positions (in units of 
        pixels) for the y and x dimensions, with column headers matching 
        `pos_cols`. Additionally, a column indicating to which (integer) time 
        index the features belong can be provided with a column header matching
        `frame_col`. This matches the output of trackpy.locate for xy or 
        from trackpy.link for xyt. The highlight color for features is `r` 
        (red) by default, but can be set on a per-feature basis by including a
        column named `stackscroller_color` in the DataFrame where each item is
        a valid color specifier for Matplotlib patch colors. The default is 
        not to highlight any features.
    pixel_aspect : tuple of float, optional
        (y,x) pixel size(ratio) for correct aspect ratio.  The default is
        `(1,1)`.
    diameter : tuple of float, optional
        (y,x) diameters for feature highlighting. The default is `(10,10)`.
    pos_cols : tuple of str, optional
        names/headers of the columns in `features` containing the positional 
        data (in units of pixels) in (y,x) order. When `features=None`, 
        `pos_cols` is ignored. The default is `(y','x')`.
    frame_col : str, optional
        name/header of the column in `features` containing the integer index
        along the time dimension in `stack` the features correspond to. When
        `features=None`, `frame_col` is ignored. The default is `frame`.
    colormap : str, optional
        matplotlib colormap name for visualising the data. The default is
        `'inferno'`.
    colormap_percentile : tuple of 2 values from 0 to 100
        lower and upper percentile of the data values to use for the min and 
        max value limits of the colormap scaling. The default is `(0.01,99.99)`
    timesteps : list of floats/ints
        list of time stamps for the video frames to display. The default is the
        frame index numbers.
    print_options : bool
        prints the keybindings and instructions how to use to the terminal. The
        default is `True`.

    Returns
    -------
    videoscroller :
        dynamic videoscroller instance that can be called with keybindings
        to update the displayed frame, when stored to a global variable.
    """
    def __init__(
            self,
            stack,
            features = None,
            pixel_aspect = (1,1),
            diameter = (10,10),
            pos_cols = ('y','x'),
            frame_col = 'frame',
            colormap = 'inferno',
            colormap_percentile = (0.01,99.99),
            timesteps = None,
            print_options = True
            ):
        #store input
        self.stack = stack
        self.shape = np.shape(self.stack)
        self.diameter = diameter
        self.pixel_aspect = pixel_aspect
        self.cmap = colormap

        #check timesteps
        if type(timesteps) == type(None):
            self.use_timesteps = False
        elif len(timesteps) != len(stack):
            raise ValueError('[Stackscroller]: number of timesteps must be '+
                             'equal to length of the stack.')
        else:
            self.use_timesteps = True
            self.timesteps = timesteps

        #set starting positions to 0
        self.x = 0
        self.y = 0
        self.t = 0
        
        #allow for the series to not start at 0
        if hasattr(stack[0], 'frame_no'):
            self.t_offset = stack[0].frame_no
        else:
            self.t_offset = 0

        #check if features are given
        if features is None:
            self.use_features = False
        #if they are, convert coords to list of numpy arrays
        else:
            self.use_features = True
            #use frame_col if it is there
            if frame_col in features.columns:
                self.features = [
                    features.loc[
                        features[frame_col]==t+self.t_offset
                    ][list(pos_cols)].to_numpy() for t in \
                        range(self.t_offset,self.t_offset+self.shape[0])
                ]
            #otherwise assume all features in first timestep
            else:
                self.features = \
                    [features[pos_cols].to_numpy()] + \
                    [features.loc[[]][pos_cols]]*(self.shape[0]-1)
            
            #check for color column, if it exists split it into timesteps
            if 'stackscroller_color' in features.columns:
                if frame_col in features.columns:
                    self.colors = [
                        features.loc[
                            features[frame_col]==t
                        ]['stackscroller_color'].to_numpy() \
                        for t in range(self.shape[0])
                    ]
                else:
                    self.colors = [features['stackscroller_color'].to_numpy()]\
                        + [[]]*(self.shape[0]-1)
            else:
                self.colors = False
                self.c = 'r'
        
        #color scaling
        self.norm = Normalize(
            vmin=np.percentile(stack,colormap_percentile[0]),
            vmax=np.percentile(stack,colormap_percentile[1])
        )

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
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)


        #start
        if print_options:
            self._print_options()
        self._update()
        plt.show(block=False)
    
    def __repr__(self):
        """interpreter string representation of stackscroller object"""
        return f"<stackscroller.videoscroller()> with shape {self.shape}"
    
    def __del__(self):
        """close figure and give warning upon garbage collection"""
        plt.close(self.fig)
        #print('closing',self)
    
    def _on_key(self,event):
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
            self._print_options()

        #update the plot
        self._update()
    
    def _print_options(self):
        """prints list of options"""
        print(' ---------------------------------------- ')
        print('|                 HOTKEYS                |')
        print('|                                        |')
        print('| left/right: move through time          |')
        print('| ctrl + arrow keys: move 10 steps       |')
        print('| f: toggle fullscreen                   |')
        print('| h: print help/hotkeys                  |')
        print(' ---------------------------------------- ')
        
    def _update(self):
        """replot the figure and features"""            
        #reset data
        self.im.set_data(self.stack[self.t])
        
        #add features
        if self.use_features:
            
            #remove old collection
            try:
                self.ec.remove()
            except:
                AttributeError
            
            #select features for current frame
            framefeatures = self.features[self.t]
            n = len(framefeatures)
            
            if self.colors:
                self.c = self.colors[self.t]
            
            #plot features onto image
            self.ec = EllipseCollection(
                [self.diameter[1]]*n,
                [self.diameter[0]]*n,
                0,
                units='xy',
                offsets = framefeatures[:,[1,0]],
                transOffset=self.ax.transData,
                edgecolors=self.c,
                facecolors='none'
            )

            self.ax.add_collection(self.ec)

        #title
        if self.use_timesteps:
            self.ax.set_title(
                'time: {:.3f} of {:.3f} s'.format(
                    self.timesteps[self.t],
                    self.timesteps[-1]
                )
            )
        elif self.t_offset != 0:
            self.ax.set_title(
                'frame {:} ({:} of {:})'.format(
                    self.t + self.t_offset,
                    self.t,
                    self.shape[0]
                )
            )
        else:
            self.ax.set_title(
                'frame {:} of {:}'.format(
                    self.t,
                    self.shape[0]
                )
            )

        #draw figure
        self.im.axes.figure.canvas.draw()
