# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:09:23 2017

@author: thomas
"""

import numpy as np

class Fold:
    """A fold model defined by angle alpha1 at inflexion point, wavelength1 and origin (ie. coordinate of first syncline)"""
    
    def __init__(self, alpha1, wavelength1,alpha2 = 0,wavelength2 = 0,b1 = 0,b2 = 0, origin = 0):
        
        self.alpha1 = alpha1
        self.wavelength1 = wavelength1
        self.alpha2 = alpha2
        self.wavelength2 = wavelength2
        self.b1 = b1
        self.b2 = b2
        self.origin = origin

class SimpleFold(Fold):
    
    def __init__(self,alpha1,wavelength1,origin):
        
        Fold.__init__(self,alpha1,wavelength1,origin)

    def Slope(self,s):
        """Define the value of the slope for a given position"""
        
        return (np.tan(self.alpha1*np.pi/180)*np.sin(2*np.pi*(s-self.origin)/self.wavelength1))

    def Value(self,s):
        """Define the shape of the fold at a given position"""
        
        return -(self.wavelength1/(2*np.pi))*np.tan(self.alpha1*np.pi/180)*(np.cos(2*np.pi*(s-self.origin)/self.wavelength1))
        
class FourierSeries():
    def __init__(self,c,wl):
        self.c = c
        self.wl = wl
    def Slope(self,s):        
        v = self.c[0]
        for w in range(len(self.wl)):
            for i in range(1,2):
                v=v + self.c[(2*i-1)+2*w]*np.cos(2*np.pi/self.wl[w] * i * (s)) + \
                self.c[(2*i)+2*w]*np.sin(2*np.pi/self.wl[w] * i * (s))
        return v
class FourierFold(Fold):
    
    def __init__(self,alpha1,wavelength1,alpha2,wavelength2,b1,b2,origin):
        
        Fold.__init__(self,alpha1,wavelength1,alpha2,wavelength2,b1,b2,origin)
        
    def Slope(self,s):
        """Define the value of the slope for a given position"""
        
        return  (self.b1*np.tan(self.alpha1*np.pi/180)*np.sin(2*np.pi*(s-self.origin)/self.wavelength1)) + (np.tan(self.alpha2*np.pi/180)*np.sin(2*np.pi*(s-self.origin)/self.wavelength2))
    
    def Value(self,s):
        """Define the shape of the fold for a given position"""
        
        return -(self.wavelength1/(2*np.pi))*self.b1*np.tan(self.alpha1*np.pi/180)*(np.cos(2*np.pi*(s-self.origin)/self.wavelength1)) - (self.wavelength2/(2*np.pi))*np.tan(self.alpha2*np.pi/180)*(np.cos(2*np.pi*(s-self.origin)/self.wavelength2))
        
        
        
    
    
class DataSet:
    
    def __init__(self,value_data_x,value_data,slope_data_x,slope_data,mixed_data_x,mixed_data_value,mixed_data_slope):
        
        self.random = False
        self.value_data_x = value_data_x
        self.value_data = value_data
        self.slope_data_x = slope_data_x
        self.slope_data = slope_data
        self.mixed_data_x = mixed_data_x
        self.mixed_data_value = mixed_data_value
        self.mixed_data_slope = mixed_data_slope
        self.UpdateDataX()
        self.nb_val_data = len(value_data_x)
        self.nb_slope_data = len(slope_data_x)
        self.nb_mixed_data = len(mixed_data_x)
        self.sigma_val = 0
        self.sigma_slope = 0
        self.sigma_val_x = 0 
        self.sigma_slope_x = 0
        self.sigma_mixed_x = 0
        self.sigma_mixed_slope = self.sigma_slope
        self.sigma_mixed_value = self.sigma_val
        
        self.sigma_val_list = []
        self.sigma_slope_list = []
        self.sigma_val_x_list = []
        self.sigma_slope_x_list = []
        self.sigma_mixed_x_list = []
        self.sigma_mixed_slope_list = []
        self.sigma_mixed_value_list = []
        
        self.TheSetter()
        
        
    def TheSetter(self):
        """Sets everything we need"""
        
        self.sigma_val_list = [self.sigma_val]*self.nb_val_data
        self.sigma_slope_list = [self.sigma_slope]*self.nb_slope_data
        self.sigma_val_x_list = [self.sigma_val_x]*self.nb_val_data
        self.sigma_slope_x_list = [self.sigma_slope_x]*self.nb_slope_data
        self.sigma_mixed_x_list = [self.sigma_mixed_x]*self.nb_mixed_data
        self.sigma_mixed_slope_list = [self.sigma_mixed_slope]*self.nb_mixed_data
        self.sigma_mixed_value_list = [self.sigma_mixed_value]*self.nb_mixed_data
        
    def RandomiseAllData(self,sigma_val_x,sigma_val,sigma_slope_x,sigma_slope,sigma_mixed_x):
        
        self.random = (sigma_val != 0) or (sigma_slope != 0) or (sigma_val_x != 0) or (sigma_slope_x != 0) or (sigma_mixed_x != 0)
        self.RandomiseValue(sigma_val)
        self.RandomiseSlope(sigma_slope)
        self.RandomiseMixed(sigma_val,sigma_slope)
        self.RandomiseXValue(sigma_val_x)
        self.RandomiseXSlope(sigma_slope_x)
        self.RandomiseXMixed(sigma_mixed_x)
        self.sigma_mixed_slope = self.sigma_slope
        self.sigma_mixed_value = self.sigma_val
        self.TheSetter()
        self.UpdateDataX()
        
    def UpdateDataX(self):
        self.no_mixed_data_x = np.append(self.value_data_x,self.slope_data_x)
        self.all_data_x = np.append(self.no_mixed_data_x,self.mixed_data_x)
        self.x_min = min(self.all_data_x)
        self.x_max = max(self.all_data_x)
                             
    def RandomiseXSlope(self,sigma):
        self.sigma_slope_x = sigma
        for i in range(self.nb_slope_data):
            self.slope_data_x[i] += np.random.randn() * sigma
        self.UpdateDataX()
                             
    def RandomiseXValue(self,sigma):
        self.sigma_val_x = sigma
        for i in range(self.nb_val_data):
            self.value_data_x[i] += np.random.randn() * sigma
        self.UpdateDataX()
                           
    def RandomiseXMixed(self,sigma):
        self.sigma_mixed_x = sigma
        for i in range(self.nb_mixed_data):
            self.mixed_data_x[i] += np.random.randn() * sigma
        self.UpdateDataX()

    def RandomiseValue(self,sigma):
        self.sigma_val = sigma
        for i in range(self.nb_val_data):
            self.value_data[i] += np.random.randn() * sigma
                           
    def RandomiseSlope(self,sigma):
        self.sigma_slope = sigma
        for i in range(self.nb_slope_data):
            self.slope_data[i] += np.random.randn() * sigma
                           
    def RandomiseMixed(self,sigma_val,sigma_slope):
        self.sigma_val = sigma_val
        self.sigma_slope = sigma_slope
        for i in range(self.nb_mixed_data):
            self.mixed_data_value[i] += np.random.randn() * sigma_val
            self.mixed_data_slope[i] += np.random.randn() * sigma_slope

class RandomDataSet(DataSet):
    
    def __init__(self, fold, x_min, x_max, nb_val_data, nb_slope_data, nb_mixed_data):
        
        # initialise fold data parameters
        self.fold = fold
        self.nb_val_data = nb_val_data
        self.nb_slope_data = nb_slope_data
        self.nb_mixed_data = nb_mixed_data
        
        value_data_x = np.zeros(nb_val_data) #random point of x coordinate
        value_data = np.zeros(nb_val_data) #"y" associated to the random x
        slope_data_x = np.zeros(nb_slope_data) #"x" asssociated to the slope
        slope_data = np.zeros(nb_slope_data) #"y" associated to the previous x
        mixed_data_x = np.zeros(nb_mixed_data) #"x" having 2 constraints (slope + value)
        mixed_data_value = np.zeros(nb_mixed_data) #"y" associated to that x
        mixed_data_slope = np.zeros(nb_mixed_data) #"slope" associated to that x
        DataSet.__init__(self,value_data_x,value_data,slope_data_x,slope_data,mixed_data_x,mixed_data_value,mixed_data_slope)
        self.InitialiseDataSet(x_min,x_max)
        
    def InitialiseDataSet(self,x_min,x_max):
        self.value_data_x = self.InitialiseX(self.value_data_x,self.nb_val_data,x_min,x_max)
        self.slope_data_x = self.InitialiseX(self.slope_data_x,self.nb_slope_data,x_min,x_max)
        self.mixed_data_x = self.InitialiseX(self.mixed_data_x,self.nb_mixed_data,x_min,x_max)
        self.GetValueData()
        self.GetSlopeData()
        self.GetMixedData()
        self.UpdateDataX()
    
    def InitialiseX(self,x_vector,n,x_min,x_max):
        for x_index in range(n):
            x_vector[x_index] = np.random.uniform(x_min,x_max)
        return np.sort(x_vector)
    
    def GetValueData(self):
        for i in range(self.nb_val_data):
            x = self.value_data_x[i]
            self.value_data[i] = self.fold.Value(x)
            
    def GetSlopeData(self):
        for i in range(self.nb_slope_data):
            x = self.slope_data_x[i]
            self.slope_data[i] = self.fold.Slope(x)
            
    def GetMixedData(self):
        for i in range(self.nb_mixed_data):
            x = self.mixed_data_x[i]
            self.mixed_data_value[i] = self.fold.Value(x)
            self.mixed_data_slope[i] = self.fold.Slope(x)


class CombinedDataSet(DataSet):
    """Regroup all the data from all the data sets into one global one"""
    
    def __init__(self,datasets):
        
        global_dataset_value_data_x = datasets[0].value_data_x
        global_dataset_value_data = datasets[0].value_data
        global_dataset_slope_data_x = datasets[0].slope_data_x
        global_dataset_slope_data = datasets[0].slope_data
        global_dataset_mixed_data_x = datasets[0].mixed_data_x
        global_dataset_mixed_data_value = datasets[0].mixed_data_value
        global_dataset_mixed_data_slope = datasets[0].mixed_data_slope
                   
        DataSet.__init__(self,global_dataset_value_data_x,global_dataset_value_data,global_dataset_slope_data_x,global_dataset_slope_data,global_dataset_mixed_data_x,global_dataset_mixed_data_value,global_dataset_mixed_data_slope)
        self.sigma_val_list = datasets[0].sigma_val_list
        self.sigma_slope_list = datasets[0].sigma_slope_list
        self.sigma_val_x_list = datasets[0].sigma_val_x_list
        self.sigma_slope_x_list = datasets[0].sigma_slope_x_list
        self.sigma_mixed_x_list = datasets[0].sigma_mixed_x_list
        self.sigma_mixed_slope_list = datasets[0].sigma_mixed_slope_list
        self.sigma_mixed_value_list = datasets[0].sigma_mixed_value_list
                                     
        self.DataCombiner(datasets)
        
        self.no_mixed_data_x = np.append(self.value_data_x,self.slope_data_x)
        self.all_data_x_list = np.append(self.no_mixed_data_x,self.mixed_data_x)
        self.x_min_all = min(self.all_data_x_list)
        self.x_max_all = max(self.all_data_x_list)
   

    def DataCombiner(self,datasets):
        """Combine the data sets"""
        
        for dataset_global_index in range(1,len(datasets)):
            self.value_data_x = np.append(self.value_data_x,datasets[dataset_global_index].value_data_x)
            self.value_data = np.append(self.value_data,datasets[dataset_global_index].value_data)
            self.slope_data_x = np.append(self.slope_data_x,datasets[dataset_global_index].slope_data_x)
            self.slope_data = np.append(self.slope_data,datasets[dataset_global_index].slope_data)
            self.mixed_data_x = np.append(self.mixed_data_x,datasets[dataset_global_index].mixed_data_x)
            self.mixed_data_value = np.append(self.mixed_data_value,datasets[dataset_global_index].mixed_data_value)
            self.mixed_data_slope = np.append(self.mixed_data_slope,datasets[dataset_global_index].mixed_data_slope)
            
            self.sigma_val_x_list  = np.append(self.sigma_val_x_list,datasets[dataset_global_index].sigma_val_x_list)
            self.sigma_val_list  = np.append(self.sigma_val_list,datasets[dataset_global_index].sigma_val_list)
            self.sigma_slope_x_list  = np.append(self.sigma_slope_x_list,datasets[dataset_global_index].sigma_slope_x_list)                                    
            self.sigma_slope_list  = np.append(self.sigma_slope_list,datasets[dataset_global_index].sigma_slope_list)
            self.sigma_mixed_x_list  = np.append(self.sigma_mixed_x_list,datasets[dataset_global_index].sigma_mixed_x_list)
            self.sigma_mixed_slope_list = np.append(self.sigma_mixed_slope_list,datasets[dataset_global_index].sigma_mixed_slope_list)
            self.sigma_mixed_value_list = np.append(self.sigma_mixed_value_list,datasets[dataset_global_index].sigma_mixed_value_list)
            