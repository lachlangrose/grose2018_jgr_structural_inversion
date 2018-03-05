# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:05:14 2017

@author: thomas
"""

from numpy import *   
from itertools import chain
from Modified_Fold import *
import Modified_Fold as mf
import numpy as np
import matplotlib.pyplot as plt


def LoadValueConstraint(file):
    x_data,y_data,slope_data = np.loadtxt(file, unpack=True)
    return x_data,y_data

def LoadSlopeConstraint(file):
    x_data,y_data,slope_data = np.loadtxt(file, unpack=True)
    return x_data,slope_data

def LoadAllConstraints(file):
    x_data,y_data,slope_data = np.loadtxt(file, unpack=True)
    return x_data,y_data,slope_data

def DataLoader(x_data,y_data,slope_data,file):
    """Load all the data needed for the simulation coming from hand typed data or file data. if there is a file, possible_file = True. Else False.If there is no file, precise "None" as file"""
    
    if file != False :
        temporary_x,temporary_y,temporary_slope = LoadAllConstraints(file)
        x_data = np.append(temporary_x,x_data)
        y_data = np.append(temporary_y,y_data)
        slope_data = np.append(temporary_slope,slope_data)
    return x_data,y_data,slope_data
        
def SortedData(x_data,y_data,slope_data,file):
    """Sort the new data list"""
    
    x_data,y_data,slope_data = DataLoader(x_data,y_data,slope_data,file)       
    x_data,y_data,slope_data = np.sort(x_data),np.sort(y_data),np.sort(slope_data)
    return x_data,y_data,slope_data

def CoefficientSaving(x_data,y_data,slope_data,file,nb_divisions,points,slope):

    coefficient_saving = FoldInterpolator(x_data,y_data,slope_data,file,nb_divisions,points,slope).CoefficientCalculation()       
    return coefficient_saving



class FoldPlot1D :  
    
    def __init__(self,datasets,nb_divisions,use_slope,points_weight = 1,slope_weight = 1,smoothing_weight = 1,folding_model_weight = 1):
        self.interpolator = FoldInterpolator(datasets,nb_divisions,use_slope,points_weight,slope_weight,smoothing_weight,folding_model_weight)
        self.datasets = datasets
        self.use_slope = use_slope
        self.nb_divisions = nb_divisions
        self.fold_model_list = []
        self.fold_model_show = False
        self.mixed_slope = True
        self.color_interpolation = 'gray'
        self.color_slope_glyphs = 'r'
        self.color_error_bar = 'k--'
        self.color_no_fold_model = 'blue'
        self.color_fold_model = 'k--'
        self.color_points_data = 'r+'
        self.color_slope_data = 'b+'
        self.color_mixed_data = 'k+'
        self.show_error_bar = True
        self.show_slope_glyphs = True
        self.show_slope_subplot = True
        self.graph_title = 'Fold Interpolation'
        self.label_x_axis = 'x coordinate'
        self.label_y_axis = 'y coordinate'
        self.data_x = []
        self.data_value = []
        self.data_slope = []
        self.slope_glyphs_y = -5
        self.glyph_size = 0.5
        self.separation_line_y = self.slope_glyphs_y + 1
        self.separation_line_color = 'k-.'
        self.position_x_legend_box = -0.2
        self.position_y_legend_box = -0.3

        
    def SlopeGlyph(self):
        """Show with a little segment the slope at a given mixed data point"""

        for data_index in range(len(self.data_x)):
            x_min_0 = self.data_x[data_index] - self.glyph_size
            x_medium = self.data_x[data_index]
            x_max_0 = self.data_x[data_index] + self.glyph_size                            
            y_0 = self.data_value[data_index] 
            
            if self.mixed_slope == True:
                sigma = self.interpolator.global_datasets.sigma_mixed_slope_list[data_index]
            
            else:
                sigma = self.interpolator.global_datasets.sigma_slope_list[data_index]
                
            slope_sigma_pos = self.data_slope[data_index]*(1 + sigma)
            h_pos = np.sqrt(slope_sigma_pos**2 + 1)
            cos_pos = 1/h_pos
            sin_pos = slope_sigma_pos/h_pos
            x_min_1_pos = x_medium + (x_min_0 - x_medium)*cos_pos
            x_max_1_pos = x_medium + (x_max_0 - x_medium)*cos_pos
            y_min_1_pos = y_0 + (x_min_0 - x_medium)*sin_pos
            y_max_1_pos = y_0 + (x_max_0 - x_medium)*sin_pos
            slope_x_sub_list_pos = [x_min_1_pos,x_max_1_pos]
            slope_data_sub_list_pos = [y_min_1_pos,y_max_1_pos]
            plt.plot(slope_x_sub_list_pos,slope_data_sub_list_pos,self.color_slope_glyphs,linewidth = 1.2)
            
            slope_sigma_neg = self.data_slope[data_index]*(1 - sigma)
            h_neg = np.sqrt(slope_sigma_neg**2 + 1)
            cos_neg = 1/h_neg
            sin_neg = slope_sigma_neg/h_neg
            x_min_1_neg = x_medium + (x_min_0 - x_medium)*cos_neg
            x_max_1_neg = x_medium + (x_max_0 - x_medium)*cos_neg
            y_min_1_neg = y_0 + (x_min_0 - x_medium)*sin_neg
            y_max_1_neg = y_0 + (x_max_0 - x_medium)*sin_neg
            slope_x_sub_list_neg = [x_min_1_neg,x_max_1_neg]
            slope_data_sub_list_neg = [y_min_1_neg,y_max_1_neg]
            
            if self.mixed_slope: 
                plt.plot(slope_x_sub_list_neg,slope_data_sub_list_neg,self.color_slope_glyphs,linewidth = 1.2) 

                
            else: 
                plt.plot(slope_x_sub_list_neg,slope_data_sub_list_neg,self.color_slope_glyphs,linewidth = 1.2) 
                
        if self.mixed_slope == False and self.use_slope == True:
            plt.plot(self.data_x,self.data_value,self.color_slope_data,label = 'Slope Data')    
            
    def ErrorBarPoints(self):
        """Error bars for the points data"""
    
        if self.datasets[0].random == False:
            return
            
        for datasets_index in range(len(self.datasets)):
            
            for inside_dataset_index in range(len(self.datasets[datasets_index].value_data_x)):
                
                #on x axis
                sigma_x = self.datasets[datasets_index].sigma_val_x
                x_medium = self.datasets[datasets_index].value_data_x[inside_dataset_index]
                y_medium = self.datasets[datasets_index].value_data[inside_dataset_index]
                sub_x_list = [x_medium - sigma_x,x_medium + sigma_x]
                sub_y_list = 2*[y_medium]
                plt.plot(sub_x_list,sub_y_list,self.color_error_bar,linewidth = 0.5)
                
                #on y axis
                sigma_value = self.datasets[datasets_index].sigma_val
                sub_x_list_bis = 2*[x_medium]
                sub_y_list_bis = [y_medium - sigma_value,y_medium + sigma_value]
                plt.plot(sub_x_list_bis,sub_y_list_bis,self.color_error_bar,linewidth = 0.5)

    def SeparationLine(self):
        """Just a line to separate the slope glyphs and the rest"""
        
        x_min = self.interpolator.global_datasets.x_min_all
        x_max = self.interpolator.global_datasets.x_max_all
        y = self.separation_line_y
        plt.plot([x_min,x_max],[y,y],self.separation_line_color)

  
    def Show(self):
        """Show data + the results of DataCalculation"""
        
        for fold_model_list_index in range(len(self.fold_model_list)):
            self.interpolator = FoldInterpolator(self.datasets,self.nb_divisions,self.use_slope,self.interpolator.points_weight,self.interpolator.slope_weight,self.interpolator.smoothing_weight,self.interpolator.folding_model_weight)
            self.interpolator.fold_model = self.fold_model_list[fold_model_list_index]
            self.interpolator.CoefficientCalculation()
            
            if self.interpolator.fold_model != False :
                plt.plot(self.interpolator.model_x,self.interpolator.X_matrix,self.color_interpolation, label = ['Fold interpolation','alpha1,wavelength1,b1,alpha2,wavelength2,b2,origin',(self.fold_model_list[fold_model_list_index].alpha1,self.fold_model_list[fold_model_list_index].wavelength1,self.fold_model_list[fold_model_list_index].b1,self.fold_model_list[fold_model_list_index].alpha2,self.fold_model_list[fold_model_list_index].wavelength2,self.fold_model_list[fold_model_list_index].b2,self.fold_model_list[fold_model_list_index].origin),'weights (p,sl,sm,f) = ',self.interpolator.points_weight,self.interpolator.slope_weight,self.interpolator.smoothing_weight,self.interpolator.folding_model_weight])
                
            else :
                plt.plot(self.interpolator.model_x,self.interpolator.X_matrix, self.color_no_fold_model, label = ['Fold interpolation (No fold model)','weights (p,sl,sm,f) = ',self.interpolator.points_weight,self.interpolator.slope_weight,self.interpolator.smoothing_weight,self.interpolator.folding_model_weight])
            
            if self.fold_model_show != False and self.fold_model_list[fold_model_list_index] != False :
                plt.plot(self.interpolator.segment_middle,self.interpolator.fold_model_shape, self.color_fold_model, label = ['Fold Model','alpha1,wavelength1,b1,alpha2,wavelength2,b2,origin',(self.fold_model_list[fold_model_list_index].alpha1,self.fold_model_list[fold_model_list_index].wavelength1,self.fold_model_list[fold_model_list_index].b1,self.fold_model_list[fold_model_list_index].alpha2,self.fold_model_list[fold_model_list_index].wavelength2,self.fold_model_list[fold_model_list_index].b2,self.fold_model_list[fold_model_list_index].origin)]) 
        
        plt.plot(self.interpolator.value_data_x,self.interpolator.value_data, self.color_points_data, label = 'Points data') 
        plt.plot(self.interpolator.mixed_data_x,self.interpolator.mixed_data_value, self.color_mixed_data, label = 'Mixed data') 
        
        if self.show_error_bar == True :
            self.ErrorBarPoints()
            
        if self.show_slope_glyphs == True :
            self.data_x = self.interpolator.mixed_data_x
            self.data_value = self.interpolator.mixed_data_value
            self.data_slope = self.interpolator.mixed_data_slope
            self.mixed_slope = True
            self.SlopeGlyph()
            
        if self.show_slope_subplot == True and self.use_slope == True:
            self.data_x = self.interpolator.slope_data_x
            self.data_value = [self.slope_glyphs_y]*len(self.interpolator.slope_data_x)
            self.data_slope = self.interpolator.slope_data
            self.mixed_slope = False
            self.SlopeGlyph()
            self.SeparationLine()
            
        plt.title(self.graph_title)
        plt.xlabel(self.label_x_axis)
        plt.ylabel(self.label_y_axis)
        plt.legend(bbox_to_anchor=(self.position_x_legend_box,self.position_y_legend_box), loc=2, borderaxespad=0.)  
        
        plt.xlabel(self.label_x_axis)
        plt.ylabel(self.label_y_axis)
        
        plt.show()
        print('Number of nodes : ',self.nb_divisions+1)
        
    def AddFoldModel(self,fold):
        self.fold_model_list = self.fold_model_list + [fold]
    


class FoldInterpolator :
    
    def __init__(self,datasets,nb_divisions,use_slope,points_weight = 1,slope_weight = 1,smoothing_weight = 1,folding_model_weight = 1) :
        
        self.use_slope = use_slope
        self.use_points = True
        self.use_relative_weight = True
        self.fold_model = False
        self.points_weight = points_weight
        self.slope_weight = slope_weight
        self.smoothing_weight = smoothing_weight
        self.folding_model_weight = folding_model_weight
        self.nb_divisions = nb_divisions
        self.nb_divisions_checker = nb_divisions
        self.data_width = 0
        self.model_x = []
        self.model_y = []
        self.subdivision_step = 0
        self.global_datasets = mf.CombinedDataSet(datasets)
        self.value_data_x = self.global_datasets.value_data_x
        self.value_data = self.global_datasets.value_data
        self.slope_data_x = self.global_datasets.slope_data_x
        self.slope_data = self.global_datasets.slope_data
        self.mixed_data_x = self.global_datasets.mixed_data_x
        self.mixed_data_value = self.global_datasets.mixed_data_value
        self.mixed_data_slope = self.global_datasets.mixed_data_slope
        self.x_data = np.append(self.value_data_x,self.mixed_data_x) 
        self.slope_or_value_x = self.x_data
        self.all_data_x = np.append(self.x_data,self.slope_data_x)
        self.y_data = np.append(self.value_data,self.mixed_data_value)
        self.slope_data_x_combined = np.append(self.slope_data_x,self.mixed_data_x)
        self.slope_data = np.append(self.slope_data,self.mixed_data_slope)   
        self.segment_middle = []
        self.fold_model_shape = []
        self.B_folding_slope_matrix = []
        self.B_folding_value_matrix = []
        self.smoothing_constraints_matrix = []
        self.points_constraints_matrix = []
        self.slope_constraints_matrix = []
        self.folding_slope_constraints_matrix = []
        self.folding_value_constraints_matrix = []
        self.combined_constraints_matrix = []
        self.squared_matrix = []
        self.leastsquare_B_matrix = []
        self.B_matrix = []
        self.X_matrix = []
        self.saving_matrix = []
        self.checker = False
        self.max_x = 0.0
        self.min_x = 0.0
        
        self.Subdivisions()

    
    def Subdivisions(self): #nb_divisions = how many little segments you want
        """ nb_divisionside in little segments the space between two points (on x) for each pair """
        
        self.model_x = []
        self.model_y = []
        if self.max_x == 0.0 and self.min_x == 0.0:
            self.max_x = max(self.all_data_x)
            self.min_x = min(self.all_data_x)
        self.data_width = abs(self.max_x - self.min_x)
        data_center = (self.max_x + self.min_x)/2.0
        #magnification_factor = 1.1
        plot_width =  self.data_width
        
        subdivision_origin = data_center - plot_width / 2.0
        self.subdivision_step = plot_width/self.nb_divisions 
        
        for i in range(self.nb_divisions+1) :
            self.model_x += [subdivision_origin + i * self.subdivision_step]
            self.model_y += [0]
    
        
    def FindSegment(self,data_index):
        """Find the segment in which x is"""    
        
        segment_number = int((self.slope_or_value_x[data_index]-self.model_x[0])//self.subdivision_step)
        return segment_number
      
    def BarycentricCoordinate (self,data_index):
        """Barycentric coordinate for a given x"""
        
        segment_number = self.FindSegment(data_index)
        barycentric_coordinate_numerator = self.slope_or_value_x[data_index]-self.model_x[segment_number]
        barycentric_coordinate_denominator = self.model_x[segment_number+1]-self.model_x[segment_number]
        barycentric_coordinate = barycentric_coordinate_numerator / barycentric_coordinate_denominator
        return barycentric_coordinate,barycentric_coordinate_denominator
    
    def DataWeight(self,sigma):
        """Calculate the weight associated to each data"""
        
        if self.use_relative_weight == True:
            return np.exp(-sigma)
        
        else:
            return 1
    
    def PointConstraintsCalculation (self,data_index):
        """Calculate the constraint for a given x"""
        
        weight = self.DataWeight(self.global_datasets.sigma_val_x)
        barycentric_coordinate = self.BarycentricCoordinate(data_index)[0]
        i_constraint = (1 - barycentric_coordinate) * weight
        i_plus_1_constraint = barycentric_coordinate * weight
        return i_constraint,i_plus_1_constraint
    
    def PointConstraintsVector(self,data_index):
        """Calculate the constraint vector associated to a given x"""
        
        segment_number = self.FindSegment(data_index)     
        point_constraints_vector = np.zeros(self.nb_divisions+1)
        i_constraint,i_plus_1_constraint = self.PointConstraintsCalculation(data_index)
        point_constraints_vector[segment_number] = i_constraint
        point_constraints_vector[segment_number+1] = i_plus_1_constraint
        return point_constraints_vector
    
    def PointConstraintsMatrix(self):
        """Calculate a constraint matrix from the constraints vectors for the points"""
        
        for data_index in range(len(self.x_data)):
            point_constraints_vector = self.PointConstraintsVector(data_index)
            self.points_constraints_matrix = np.append(self.points_constraints_matrix,point_constraints_vector)

        self.points_constraints_matrix = self.points_constraints_matrix.reshape(len(self.x_data),self.nb_divisions+1)
        return self.points_constraints_matrix * self.points_weight

    def SlopeConstraintsCalculation(self,data_index):
        """Calculate the constraint for the slope"""
        
        sigma = self.DataWeight(self.global_datasets.sigma_slope_x)
        barycentric_coordinate_denominator = self.BarycentricCoordinate(data_index)[1]
        i_constraint = (-1/barycentric_coordinate_denominator)*sigma
        i_plus_1_constraint = (1/barycentric_coordinate_denominator)*sigma
        return i_constraint,i_plus_1_constraint
    
    def SlopeConstraintsVector(self,data_index):
        """Calculate the constraint vector associated to a slope"""

        segment_number = self.FindSegment(data_index)     
        slope_constraints_vector = np.zeros(self.nb_divisions+1)         
        i_constraint,i_plus_1_constraint = self.SlopeConstraintsCalculation(data_index)
        slope_constraints_vector[segment_number] = i_constraint
        slope_constraints_vector[segment_number+1] = i_plus_1_constraint
        return slope_constraints_vector
    
    def SlopeConstraintsMatrix(self):
        """Calculate a constraint matrix from the constraints vectors for the slopes"""
        
        self.slope_or_value_x = self.slope_data_x_combined
        for data_index in range(len(self.slope_data_x_combined)):
            slope_constraints_vector = self.SlopeConstraintsVector(data_index)
            self.slope_constraints_matrix = np.append(self.slope_constraints_matrix,slope_constraints_vector)
        self.slope_constraints_matrix = self.slope_constraints_matrix.reshape(len(self.slope_data_x_combined),self.nb_divisions+1)
        return self.slope_constraints_matrix*self.slope_weight
        
    def SmoothingConstraintsCalculation(self,node_index):
        """Calculate the constraints linked to the nodes (minimize the slope differences between each points/nodes)"""

        i_less_1_constraint = (self.model_x[node_index]-self.model_x[node_index+1])/((self.model_x[node_index]-self.model_x[node_index-1])*(self.model_x[node_index+1]-self.model_x[node_index]))
        i_constraint = (self.model_x[node_index+1]-self.model_x[node_index-1])/((self.model_x[node_index]-self.model_x[node_index-1])*(self.model_x[node_index+1]-self.model_x[node_index]))
        i_plus_1_constraint = (self.model_x[node_index-1]-self.model_x[node_index])/((self.model_x[node_index]-self.model_x[node_index-1])*(self.model_x[node_index+1]-self.model_x[node_index]))
        smoothing_constraint = i_less_1_constraint,i_constraint,i_plus_1_constraint
        return  smoothing_constraint 

    def SmoothingConstraintsVector(self,node_index):
        """Calculate the constraints vector associated to the smoothing step"""
        
        smoothing_constraints_vector = np.zeros(self.nb_divisions+1)
        smoothing_constraints_vector[node_index-1] = self.SmoothingConstraintsCalculation(node_index)[0]
        smoothing_constraints_vector[node_index] = self.SmoothingConstraintsCalculation(node_index)[1]                            
        smoothing_constraints_vector[node_index+1] = self.SmoothingConstraintsCalculation(node_index)[2]
    
        return smoothing_constraints_vector 
        
    def SmoothingConstraintsMatrix(self):
        """Calculate the constraints matrix concerning the smoothing step"""
        
        for node_index in range(1,self.nb_divisions):
            smoothing_constraints_vector = self.SmoothingConstraintsVector(node_index)
            self.smoothing_constraints_matrix = np.append(self.smoothing_constraints_matrix,smoothing_constraints_vector)
        self.smoothing_constraints_matrix = self.smoothing_constraints_matrix.reshape(self.nb_divisions-1,self.nb_divisions+1)      
        return self.smoothing_constraints_matrix*self.smoothing_weight
    
    def FoldingSlopeConstraintsCalculation(self,node_index):
        """Calculate the constraints linked to a given folding model concerning slope part"""
        
        i_constraint = -1/(self.model_x[node_index+1]-self.model_x[node_index])
        i_plus_1_constraint = 1/(self.model_x[node_index+1]-self.model_x[node_index])
        return i_constraint,i_plus_1_constraint

    def FoldingSlopeConstraintsVector(self,node_index):
        """Calculate the constraints vector associated to the folding model concerning slope part"""
        
        folding_slope_constraints_vector = np.zeros(self.nb_divisions+1)
        i_constraint,i_plus_1_constraint = self.FoldingSlopeConstraintsCalculation(node_index)
        folding_slope_constraints_vector[node_index] = i_constraint
        folding_slope_constraints_vector[node_index+1] = i_plus_1_constraint 
        return folding_slope_constraints_vector

    def FoldingSlopeConstraintsMatrix(self):
        """Calculates the constraints matrix concerning the folding model concerning slope part"""
        
        for node_index in range(self.nb_divisions):
            folding_slope_constraints_vector = self.FoldingSlopeConstraintsVector(node_index)
            self.folding_slope_constraints_matrix = np.append(self.folding_slope_constraints_matrix,folding_slope_constraints_vector)
        self.folding_slope_constraints_matrix = self.folding_slope_constraints_matrix.reshape(self.nb_divisions,self.nb_divisions+1)
        return self.folding_slope_constraints_matrix*self.folding_model_weight
    
   
    def FoldingSlopeBMatrix(self):
        """Calculates the part of the B matrix associated to the folding constraints"""
        
        for segment_number in range(self.nb_divisions):
            self.segment_middle = np.append(self.segment_middle,(self.model_x[segment_number+1]+self.model_x[segment_number])/2)
            
        for middle_index in range(len(self.segment_middle)):
           self.B_folding_slope_matrix = np.append(self.B_folding_slope_matrix,self.fold_model.Slope(self.segment_middle[middle_index]))         
        return self.B_folding_slope_matrix
    
    
    def SigmaValue(self):
        """Calculates the sigma value"""
         
        sigma_value = []
        for sigma_value_list_index in range(len(self.global_datasets.sigma_val_list)):
            sigma_value = np.append(sigma_value,self.DataWeight(self.global_datasets.sigma_val_list[sigma_value_list_index]))    
        
        for sigma_mixed_value_list_index in range(len(self.global_datasets.sigma_mixed_value_list)):
            sigma_value = np.append(sigma_value,self.DataWeight(self.global_datasets.sigma_mixed_value_list[sigma_mixed_value_list_index]))
        
        return sigma_value
    
    def SigmaSlope(self):
        """Calculates the sigma value"""
         
        sigma_slope = []
        for sigma_slope_list_index in range(len(self.global_datasets.sigma_slope_list)):
            sigma_slope = np.append(sigma_slope,self.DataWeight(self.global_datasets.sigma_slope_list[sigma_slope_list_index]))    
        
        for sigma_mixed_slope_list_index in range(len(self.global_datasets.sigma_mixed_slope_list)):
            sigma_slope = np.append(sigma_slope,self.DataWeight(self.global_datasets.sigma_mixed_slope_list[sigma_mixed_slope_list_index]))
        
        return sigma_slope
        
    
                
    def AssembleSystem(self):
        """The final matrix, with all three constraints combined + calculate associated B_matrix"""

        zeros_vector = (self.nb_divisions-1)*[0]
        sigma_value = self.SigmaValue()
        sigma_slope = self.SigmaSlope()
        
        if self.use_points == True:
            self.combined_constraints_matrix =  np.append(self.combined_constraints_matrix,self.PointConstraintsMatrix())
            self.B_matrix = np.append(self.B_matrix,sigma_value*[self.y_data*self.points_weight])
        
        if self.use_slope == True :
           self.combined_constraints_matrix = np.append(self.combined_constraints_matrix,self.SlopeConstraintsMatrix())
           self.B_matrix = np.append(self.B_matrix,sigma_slope*[self.slope_data*self.slope_weight])
           
        if self.fold_model != False :
           
           self.combined_constraints_matrix = np.append(self.combined_constraints_matrix,self.FoldingSlopeConstraintsMatrix())
           self.B_matrix = np.append(self.B_matrix,self.FoldingSlopeBMatrix()*self.folding_model_weight)
           
        self.combined_constraints_matrix = np.append(self.combined_constraints_matrix,self.SmoothingConstraintsMatrix())
        self.combined_constraints_matrix = self.combined_constraints_matrix.reshape(int(len(self.combined_constraints_matrix)/(self.nb_divisions+1)),self.nb_divisions+1)
        self.B_matrix = np.append(self.B_matrix,zeros_vector)
        self.B_matrix = self.B_matrix.reshape(len(self.B_matrix),1)
        
    def NbDivisionsChangementChecker(self):
        """Check if the nb_divisions is changed by the user"""
        
        if self.nb_divisions != self.nb_divisions_checker:
            self.checker = True
        
        return self.checker
        
    
    def CoefficientCalculation(self):
        """To find the coefficient needed to draw the final curve"""
        
        self.checker = self.NbDivisionsChangementChecker()
        if self.checker == True:
            self.model_x = []
            self.model_y = []
            self.segment_middle = []
            self.smoothing_constraints_matrix = []
            self.points_constraints_matrix = []
            self.slope_constraints_matrix = []
            self.combined_constraints_matrix = []
            self.folding_slope_constraints_matrix = []
            self.folding_value_constraints_matrix = []
            self.B_folding_slope_matrix = []
            self.B_folding_value_matrix = []
            self.B_matrix = []
            self.squared_matrix = []
            self.leastsquare_B_matrix = []

        # initialize result vector
        self.X_matrix = []
        
        # build least square system
        self.AssembleSystem()
        self.BuildLeastSquareSystem()
        
        # solve
        self.X_matrix = np.linalg.solve(self.squared_matrix,self.leastsquare_B_matrix)
        return self.X_matrix
    
    def BuildLeastSquareSystem(self):
        """Multiply the matrix and the combined constraints transposed matrix"""
        
        self.transposed_combined_constraints_matrix = np.array((self.combined_constraints_matrix)).T
        self.squared_matrix = np.dot(self.transposed_combined_constraints_matrix,self.combined_constraints_matrix) 
        self.leastsquare_B_matrix = np.dot(self.transposed_combined_constraints_matrix,self.B_matrix)
        
    def CoefficientsSaving(self):
        """To save the X_matrix"""
        
        if self.saving_matrix == [] :
            self.saving_matrix = [list(chain.from_iterable(self.X_matrix))]
        
        elif self.saving_matrix != [] and list(chain.from_iterable(self.X_matrix)) != self.saving_matrix[len(self.saving_matrix)-1]: 
            self.saving_matrix += [list(chain.from_iterable(self.X_matrix))]
            
        return self.saving_matrix
    
    
        
                
                
        
