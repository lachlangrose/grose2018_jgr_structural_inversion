import matplotlib.pyplot as plt
import numpy as np
class fold:
    def __init__(self, alpha, wavelength, origin):
        self.alpha = alpha
        self.wavelength = wavelength
        self.origin = origin
    def pos(self,s):
       return(np.tan(self.alpha*0.0174532925)*np.sin(2*np.pi*(s-self.origin)/self.wavelength))
class fourierFold:
    def __init__(self, b1, alpha1,wavelength1,b2,alpha2,wavelength2,origin):
        self.b1 = b1
        self.alpha1 = alpha1
        self.wavelength1 = wavelength1
        self.b2 = b2
        self.alpha2 = alpha2
        self.wavelength2 = wavelength2
        self.origin = origin
    def pos(self,s):
        return self.b1*np.tan(self.alpha1*np.pi/180)*np.sin(2*np.pi*(s-self.origin)/self.wavelength1)+self.b2*np.tan(self.alpha2*np.pi/180)*np.sin(2*np.pi*(s-self.origin)/self.wavelength2)
class foldRotation:
    def __init__(self, fold):
        self.fold = fold        
    def angle(self,s):
        return (180/np.pi)*np.arctan(self.fold.pos(s))
    def pos(self,s):
        return self.fold.pos(s)
class foldFigure:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1,sharex=True,figsize=(32, 12),
                           subplot_kw={'axisbg':'#EEEEEE',
                                       'axisbelow':True})
        self.x = np.linspace(0,100,100)
        self.ax.set_xlim(0, 105)
      #  self.ax[2].set_xlim(0, 200)
        self.ax.set_ylim(-90, 90)
        self.ax.grid(color='w', linewidth=.1, linestyle='solid')
    def add_fold_rotation_curve(self,fold, colour):
        self.ax.plot(self.x,fold.angle(self.x),color = colour)
        #self.ax[0].plot.set_title('Foliation Scalar')
        
        self.ax[1].plot(0)
        self.ax[1].plot(self.x,fold.pos(self.x),color=colour, lw = 1, alpha=0.4)
    def add_xy(self,xv,yv,colour):
        self.ax.plot(xv,yv,colour)

    def get_fig(self):
        return fig
    def add_fig_title(self,title,size):
        self.ax.set_title(title,fontsize=size,fontweight='bold')
        self.ax.set_xlabel('Cartesian Distance || Foliation Scalar Field',fontsize=size-5)
        self.ax.set_ylabel('Fold Rotation Angle',fontsize=size-5)
        self.ax.yaxis.grid(b=True, which='major', color='b', linestyle='--')
        self.ax.tick_params(axis='x',labelsize=size-8)
        self.ax.tick_params(axis='y',labelsize=size-8)
    def save_figure(self,name):
        plt.savefig(name)
def shake_xy(x,y,noiseper):
    y = np.tan(y*np.pi/180.0)
    newx = x #+ np.random.normal(-noiseper,noiseper,len(x))/100*x
    newy = y + np.random.normal(-noiseper/2.,noiseper/2.,len(x))/100*y
    newy = np.arctan(newy)*180.0/np.pi
    return newx, newy
