"""
To implement the zero temperature imaginary evolution one would need:
1. efficient representation of the Gamma-correlator and F-correlator:
    1.1 first a one-to-one mapping from basis_label to basis_id
    1.2 then express the Gamma-correlator and F-correlator as sparse matrices
    alternatively, sparate basis_label to sym_label and intra_label,
    express the correlators as dense matrices within each symmetry sector
2. imaginary time evolution of the master equation
    2.1 first an initialization respect given symmetry
    2.2 stable evolution with error control
3. post processing
4. parallelization 
"""
import numpy as np
from typing import List, Tuple, Dict
from numpy.linalg import matmul, norm
from numpy import sqrt, exp
import time

class Cell:
    def __init__(self, dim:int, vec_car:np.ndarray):
        self.dim = dim
        if vec_car.shape != (dim, dim):
            raise ValueError("incompatible cell dimensions!")
        else:
            self.vec_car = vec_car
    def volume(self):
        if self.dim == 1:
            return np.abs(self.vec_car[0][0])
        elif self.dim == 2:
            return np.abs(self.vec_car[0][0]*self.vec_car[1][1]-self.vec_car[0][1]*self.vec_car[1][0])
        elif self.dim == 3:
            return np.abs(matmul(self.vec_car[2], np.cross(self.vec_car[0], self.vec_car[1])))
    def gvec_car(self) -> np.ndarray:
        if self.dim == 1:
            return 2*np.pi/self.volume()
        elif self.dim == 2:
            return np.array([[self.vec_car[1][1], -self.vec_car[1][0]],
                             [-self.vec_car[0][1], self.vec_car[0][0]]])*2*np.pi/self.volume()
        elif self.dim == 3:
            return np.array([np.cross(self.vec_car[1], self.vec_car[2]),
                            np.cross(self.vec_car[2], self.vec_car[0]),
                            np.cross(self.vec_car[0], self.vec_car[1])])*2*np.pi/self.volume()

def timing(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"runtime = {end_time - start_time}")
    return result

def functional_if(crit, true_return, false_return=None):
    if crit:
        return true_return
    else:
        return false_return
    
def kronecker_delta(i,j):
    if i==j:
        return 1
    else:
        return 0

def diagonal_mat(diag_ele:np.ndarray) -> np.ndarray:
    length = len(diag_ele)
    mat = np.zeros((length, length))
    for i in range(length):
        mat[i,i] = diag_ele[i]
    return mat

class PackedTuple:
    def __init__(self, val):
        self.val = val

def multidim_list(ranges:List[Tuple], flatten = False, function=tuple) -> List:
    '''
    generate a multi-dimensional list by inserting arguments in 'ranges' to 'function'
    '''
    def inner_multidim_list(val:List, ranges:List[Tuple]) -> List:
        if len(ranges) == 0 and flatten:
            return PackedTuple(tuple(val))
        elif len(ranges) == 0 and not flatten:
            return function(tuple(val))
        current_range = ranges[0]
        if hasattr(current_range, '__len__'):
            return [inner_multidim_list(val+[i], ranges[1:]) for i in range(*current_range)]
        elif isinstance(current_range, int):
            return [inner_multidim_list(val+[i], ranges[1:]) for i in range(current_range)]
    if flatten:
        arr_tem = np.array(inner_multidim_list([],ranges)).flatten()
        return [function(item.val) for item in arr_tem]
    else:
        return inner_multidim_list([],ranges)

#print(multidim_list(((1,3),2,2), flatten=True))

class ReciproLabel:
    """
    immutable, a label for reciprocal vector
    """
    def __init__(self, k:tuple):
        self._k = tuple(k)
    @property
    def kint(self):
        return np.array(self._k)
    
    def __eq__(self, other):
        if isinstance(other, ReciproLabel) and self._k == other._k:
            return True
        else:
            return False
    def __hash__(self):
        return hash(self._k)
    def __str__(self):
        return f"Recipro{self._k}"
    def __repr__(self):
        return f"Recipro{self._k}"
    def kadd(self, k_add:tuple):
        return ReciproLabel(self.kint + np.array(k_add))
    
class BasisLabel(ReciproLabel):
    """
    immutable, a label for reciprocal basis
    """
    def __init__(self, k:tuple, spin:int = None, orbit:int = None):
        super().__init__(k)
        self.spin = spin
        self.orbit = orbit
    def __eq__(self, other):
        if isinstance(other, BasisLabel) and self._k == other._k and self.spin == other.spin and self.orbit == other.orbit:
            return True
        else:
            return False
    def __hash__(self):
        return hash((self._k, self.spin, self.orbit))
    def __str__(self):
        return f"Recipro{self._k}&spin{self.spin}&orbit{self.orbit}"
    def __repr__(self):
        return f"Recipro{self._k}&spin{self.spin}&orbit{self.orbit}"
    def kadd(self, k_add:tuple):
        return BasisLabel(self.kint + np.array(k_add), self.spin, self.orbit)


class ReciproDict:
    '''
    a bi-dictionary which provides a bijection between bases and ids for each bloch_k
    '''
    def __init__(self, val:Dict[ReciproLabel,List[ReciproLabel]]):
        self._val = val
        self._id = {bloch_k: {self._val[bloch_k][i]:i for i in range(len(self._val[bloch_k]))} for bloch_k in self._val}
        self.bloch_list = list(self._val.keys())
    def get_size(self, bloch_k:ReciproLabel) -> int:
        return len(self._val[bloch_k])
    def get_id(self, bloch_k:ReciproLabel, latis_g:ReciproLabel) -> int:
        return self._id[bloch_k][latis_g]
    def get_val(self, bloch_k:ReciproLabel, latis_g_id:int) -> BasisLabel:
        return self._val[bloch_k][latis_g_id]


class SuperCell:
    '''
    specify a continuum supercell with periodic boundary condition, including the unitcell('unitcell'), supercell size('sc_size'), 
    whether or not the system has spin('spinfull'),
    and number of orbitals in a unitcell('num_orbits')
    '''
    def __init__(self, dim:int, unitvec_car:np.ndarray, sc_size:Tuple, spinfull=False, num_orbits=1):
        self.unitcell = Cell(dim, unitvec_car)
        if not hasattr(sc_size, "__len__"):
            self.sc_size = tuple(sc_size for _ in range(dim))
        elif len(sc_size) == dim:
            self.sc_size = tuple(sc_size)
        else:
            raise ValueError("incompatible cell dimensions!")
        self.ktocar = matmul(np.transpose(self.unitcell.gvec_car()), np.linalg.inv(diagonal_mat(self.sc_size)))
        self.spinfull = spinfull
        self.num_orbits = num_orbits
    @property
    def dim(self) -> int:
        return self.unitcell.dim
    def kcar(self, klabel) -> np.ndarray:
        if isinstance(klabel, ReciproLabel):
            return matmul(self.ktocar, klabel.kint)
        elif isinstance(klabel, BasisLabel):
            return matmul(self.ktocar, klabel.kint)
        else:
            return matmul(self.ktocar, np.array(klabel)) #treat klabel as a list or tuple or array
    def generate_bloch_list(self) -> List[ReciproLabel]:
        '''
        generate a kgrid with in the unitcell spanned by {a_i}
        '''
        return multidim_list(self.sc_size, flatten=True, function=ReciproLabel)
    def generate_recipro_dict(self, cutoff_norm) -> ReciproDict:
        bloch_list = self.generate_bloch_list()
        radius = int(np.ceil(sqrt(cutoff_norm**2/min(np.linalg.eigvalsh(matmul(np.transpose(self.ktocar), self.ktocar))))))
        if not self.spinfull and self.num_orbits == 1:    
            basis_ranges = [(-int(np.floor((radius+self.sc_size[i])/self.sc_size[i]))*self.sc_size[i],
                                int(np.floor(radius/self.sc_size[i]) + 1)*self.sc_size[i], self.sc_size[i]) 
                            for i in range(self.dim)]
            target_func = ReciproLabel
        elif self.spinfull and self.num_orbits != 1:
            basis_ranges = [(0,2), (0,self.num_orbits)] + [(-int(np.floor((radius+self.sc_size[i])/self.sc_size[i]))*self.sc_size[i],
                                int(np.floor(radius/self.sc_size[i]) + 1)*self.sc_size[i], self.sc_size[i]) 
                            for i in range(self.dim)]
            def target_func(arg:tuple):
                return BasisLabel(arg[2:], spin=arg[0], orbit=arg[1])
        else:
            raise ValueError("supercell with either spin or orbit has not been implemented yet!")
        basis_candidates = multidim_list(basis_ranges, flatten=True, function=target_func)
        val = {bloch_k:[latis_g for latis_g in basis_candidates 
                                if norm(self.kcar(bloch_k) + self.kcar(latis_g)) < cutoff_norm]
                        for bloch_k in bloch_list}
        return ReciproDict(val)

def construct_h0(supercell:SuperCell, recipro_dict:ReciproDict, bloch_k:ReciproLabel) -> np.ndarray:
    sector_size = recipro_dict.get_size(bloch_k)
    sc_size = supercell.sc_size
    def h0_element(i:int, j:int):
        basis_i = recipro_dict.get_val(bloch_k, i)
        basis_j = recipro_dict.get_val(bloch_k, j)
        ################ begin parameters
        v = 15
        phi = np.pi*140/180
        w = -13
        me = -0.43*1.5*10**(-3)
        delta = 0.0
        h = 0.0
        ################ end parameters
        gv = [(sc_size[0],0),(0,sc_size[1]),(-sc_size[0],sc_size[1]),
              (-sc_size[0],0),(0,-sc_size[1]),(sc_size[0],-sc_size[1]),(0,0)]
        kp = 1/3*(-2*np.array(gv[0]) + np.array(gv[1])) #in direct coordinates
        km = 1/3*(-np.array(gv[0]) - np.array(gv[1])) #in direct coordinates
        rlt_k = tuple(basis_i.kint - basis_j.kint)
        isgv = [rlt_k == gv[i] for i in range(len(gv))]
        if basis_i.spin == basis_j.spin == 0:
            if basis_i.orbit == 0 and basis_j.orbit == 0:
                return functional_if(isgv[-1], 1/2/me*norm(supercell.kcar(bloch_k) + supercell.kcar(basis_i) - supercell.kcar(kp))**2, 0)+\
                        functional_if(isgv[0] or isgv[2] or isgv[4], v*exp(1j*phi), 0) +\
                        functional_if(isgv[1] or isgv[3] or isgv[5], v*exp(-1j*phi), 0) + delta/2 + h/2
            elif basis_i.orbit == 0 and basis_j.orbit == 1:
                return functional_if(isgv[1] or isgv[2] or isgv[-1], w, 0)
            elif basis_i.orbit == 1 and basis_j.orbit == 0:
                return functional_if(isgv[4] or isgv[5] or isgv[-1], w, 0)
            elif basis_i.orbit == 1 and basis_j.orbit == 1:
                return functional_if(isgv[-1], 1/2/me*norm(supercell.kcar(bloch_k) + supercell.kcar(basis_i) - supercell.kcar(km))**2, 0)+\
                        functional_if(isgv[0] or isgv[2] or isgv[4], v*exp(-1j*phi), 0) +\
                        functional_if(isgv[1] or isgv[3] or isgv[5], v*exp(1j*phi), 0) - delta/2 + h/2
            else:
                raise ValueError("problematic orbit value!")
        elif basis_i.spin == basis_j.spin == 1:
            if basis_i.orbit == 0 and basis_j.orbit == 0:
                return functional_if(isgv[-1], 1/2/me*norm(supercell.kcar(bloch_k) + supercell.kcar(basis_i) - supercell.kcar(km))**2, 0)+\
                        functional_if(isgv[0] or isgv[2] or isgv[4], v*exp(1j*phi), 0) +\
                        functional_if(isgv[1] or isgv[3] or isgv[5], v*exp(-1j*phi), 0) + delta/2 - h/2
            elif basis_i.orbit == 0 and basis_j.orbit == 1:
                return functional_if(isgv[4] or isgv[5] or isgv[-1], w, 0)
            elif basis_i.orbit == 1 and basis_j.orbit == 0:
                return functional_if(isgv[1] or isgv[2] or isgv[-1], w, 0)
            elif basis_i.orbit == 1 and basis_j.orbit == 1:
                return functional_if(isgv[-1], 1/2/me*norm(supercell.kcar(bloch_k) + supercell.kcar(basis_i) - supercell.kcar(kp))**2, 0)+\
                        functional_if(isgv[0] or isgv[2] or isgv[4], v*exp(-1j*phi), 0) +\
                        functional_if(isgv[1] or isgv[3] or isgv[5], v*exp(1j*phi), 0) - delta/2 - h/2
            else:
                raise ValueError("problematic orbit value!")
        else:
            return 0
    return np.array([[h0_element(i,j) for j in range(sector_size)] for i in range(sector_size)])




if __name__ == "__main__":
    sc_size = (4, 4)
    supercell = SuperCell(2, np.array([[1,0],[-1/2, np.sqrt(3)/2]])*180/(4*np.pi), sc_size, spinfull=True, num_orbits=2)
    dk = norm(supercell.ktocar[:,0])
    # print(dk)
    # print(supercell.dim)
    # print(supercell.sc_size)
    # print(supercell.ktocar)
    recipro_dict = supercell.generate_recipro_dict(dk*4.1)
    bloch_list = recipro_dict.bloch_list
    # for bloch in bloch_list:
    #     print(f"bloch vector: {bloch}")
    #     sector_size = recipro_dict.get_size(bloch)
    #     print(sector_size)
    #     print([recipro_dict.get_val(bloch,i) for i in range(sector_size)])
    #print(construct_h0(supercell, recipro_dict, ReciproLabel((0,0))))
    h0 = timing(construct_h0, supercell, recipro_dict, ReciproLabel((0,0)))
    print(timing(np.linalg.eigvalsh, h0))
    #print(np.linalg.eigvalsh(construct_h0(supercell, recipro_dict, ReciproLabel((0,0)))))