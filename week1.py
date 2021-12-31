#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
	#return a numpy array with one at all index
	array=None
	#TODO
	array=np.ones(shape)
	return array			


#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
	#return a numpy array with zeros at all index
	array=None
	#TODO
	array=np.zeros(shape)
	return array

#input: int  
def create_identity_numpy_array(order):
	#return a identity numpy array of the defined order
	array=None
	#TODO
	array=np.identity(order,dtype=int)
	return array


#input: numpy array
def matrix_cofactor(array):
	#return cofactor matrix of the given array
	a=None
	a=np.linalg.inv(array).T*np.linalg.det(array)
	#TODO
	return array

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
	#note: shape is of the forst (x1,x2)
	#return D1 x (X1 ** coef1) + D2 x (X2 ** coef2) +b
	# where D1 is random matrix of shape shape1 with seed1
	# where D2 is random matrix of shape shape2 with seed2
	# where B is a random matrix of comaptible shape with seed3
	# if dimension mismatch occur return -1
	ans=None
	#TODO
	np.random.seed(seed1)
	D1 = np.random.rand(shape1[0],shape1[1])
	np.random.seed(seed2)
	D2 = np.random.rand(shape2[0], shape2[1])
	
	
	if(shape1[1] == X1.shape[0]):
	        halfans1 = np.matmul(D1, np.power(X1,coef1))
	else:
	        return -1
	        
	if(shape2[1] == X2.shape[0]):
	        halfans2 = np.matmul(D2, np.power(X2,coef2))
	else:
	        return -1
	        
	if(halfans1.shape == halfans2.shape):
	        np.random.seed(seed3)
	        B = np.random.rand(halfans1.shape[0], halfans2.shape[1])
	        ans = halfans1 + halfans2 + B
	else:
	        return -1
	return ans


def fill_with_mode(filename, column):
	df=pd.read_csv(filename)
	df[column].fillna(df[column].mode()[0],inplace=True)
	return df


def fill_with_group_average(df, group, column):
	df[column].fillna(df.groupby(group)[column].transform('mean'), inplace=True)
	return df


def get_rows_greater_than_avg(df, column):
	df=df.loc[df[column]>df[column].mean(skipna=True)]
	return df
