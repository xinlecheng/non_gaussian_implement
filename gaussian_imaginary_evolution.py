"""
To implement the zero temperature imaginary evolution one would need:
1. efficient representation of the Gamma-correlator and F-correlator:
    1.1 first a one-to-one mapping from basis_label to basis_id
    1.2 then express the Gamma-correlator and F-correlator as sparse matrices
2. imaginary time evolution of the master equation
    2.1 first an initialization respect given symmetry
    2.2 stable evolution with error control
3. post processing
4. parallelization 
"""
import numpy as np
from typing import List, Tuple, Dict
from numpy.linalg import matmul

class Cell:
    def __init__(self, dim:int, vec_car:np.ndarray):
        self.dim = dim
        if vec_car.shape != (dim, dim):
            raise ValueError("incompatible cell dimensions!")
        else:
            self.vec_car = vec_car
    def volume(self):
        if self.dim == 2:
            return np.abs(self.vec_car[0][0]*self.vec_car[1][1]-self.vec_car[0][1]*self.vec_car[1][0])
        elif self.dim == 3:
            return np.abs(matmul(self.vec_car[2], np.cross(self.vec_car[0], self.vec_car[1])))
    def gvec_car(self) -> np.ndarray:
        if self.dim == 2:
            return np.array([[self.vec_car[1][1], -self.vec_car[1][0]],
                             [-self.vec_car[0][1], self.vec_car[0][0]]])*2*np.pi/self.volume()
        elif self.dim == 3:
            return np.array([np.cross(self.vec_car[1], self.vec_car[2]),
                            np.cross(self.vec_car[2], self.vec_car[0]),
                            np.cross(self.vec_car[0], self.vec_car[1])])*2*np.pi/self.volume()

class ReciproLabel:
    """
    immutable, a label for the basis, here we label by the (bloch_k, latis_g) combination
    """
    def __init__(self, bloch_k, latis_g):
        pass

if __name__ == "__main__":
    cell = Cell(2, np.array([[1,0],[-1/2,np.sqrt(3)/2]]))
    print(cell.gvec_car())