'''
Created on Aug 24, 2017

@author: sandeep.singh
'''
import numpy as np
x = np.array([1,2,3], dtype=np.int64)
x
array([1, 2, 3])
x.dtype
dtype('int64')
y = x.astype(np.int32)
y
array([1, 2, 3], dtype=int32)