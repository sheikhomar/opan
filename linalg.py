import numpy as np

class LinearSystem:
    def __init__(self, A, b):
        self._A = np.array(A)
        
        # Convert b to a column vector
        self._b = np.array(b).reshape(-1, 1)
        
        # Append b as a column to A
        self._Ab = np.append(self._A, self._b, axis=1)

    def relative_residual(self, point):
        point_vec = np.array(point).reshape(-1, 1)
        #point_vec = np.array(point).T
        numerator = np.linalg.norm(self._b - np.dot(self._A, point_vec))
        denominator = np.linalg.norm(self._b)
        return numerator / denominator
   
    def least_squares_solution(self):
        """
        Computes the least-squares solution of the linear system.
        """
        A = self._A
        
        # Compute A^T A
        A_t_A = A.T @ A
        
        # Find the inverse
        inv_A_t_A = np.linalg.inv(A_t_A)
        
        # Multiply with A^T
        m = inv_A_t_A @ A.T
        
        # Multiply with b
        return m @ self._b
    
    def solve_minimum_norm(self):
        """
        Find a solution to a linear system with minimum norm i.e. 
        minimise ||x|| subject to Ax=b
        """
        # 
        A = self._A
        b = self._b
        return A.T @ np.linalg.inv(A @ A.T) @ b
    
    def rank(self):
        """
        Computes the rank of the system i.e. rank [A|b].
        
        The rank of a matrix is the number of independent rows or columns of that matrix.
        """
        return np.linalg.matrix_rank(self._Ab)
    
    def rankA(self):
        """
        Computes the rank of the matrix A i.e. rank(A).
        
        The rank of a matrix is the number of independent rows or columns of that matrix.
        """
        return np.linalg.matrix_rank(self._A)

    def is_inconsistent(self):
        """
        Determines whether the system is inconsistent i.e., whether the vector b belongs to the range of A.
        
        The range (aka. image or column space) of a matrix is the the span 
        (set of all possible linear combinations) of its column vectors.
        
        A system is inconsistent if the rank(A) < rank([A|b]).
        """
        return self.rankA() < self.rank()
