

class Derivatives:

    def __init__(self, vals, grid1, grid2=None, fd=None):
        '''
        This is a class for calculating derivatives of functions from an evenly spaced grid in 1 or 2 dimensions. It
        employs either finite difference or the derivative methods from the Colbert Miller paper on DVR and a paper from
        Anne and Mark Johnson about some radical H4O2+
        :param vals: The values of the eigen functions from a DVR
        :type vals: np array (1-2 dimensions)
        :param grid1: The grid that the values are calculated from
        :type grid1: np array (1 dimension)
        :param grid2: If the eigen function is 2d then this is the grid in the second dimension
        :type grid2: np array (1 dimension)
        :param fd: A parameter to indicate if finite difference will be used
        :type fd: True or None
        '''
        self.vals = vals
        self.fd = fd
        self.grid1 = grid1
        self.grid2 = grid2

    @staticmethod
    def first_derivative(grid):
        '''
        This function returns the matrix to dot into your eigen function to return the first derivative using eq A6 from
        DOI:10.1021/jp811493s
        :param grid: The grid that the values are calculated from
        :type grid: np array (1 dimension)
        :return: A matrix that when dotted into the values of your function, will return the derivative of that function
        :rtype: np array (2 dimensions)
        '''
        import numpy as np
        N = len(grid)
        a = grid[0]
        b = grid[-1]
        dx = 1 / ((float(N)) / (b - a))

        Tii = np.zeros(N)

        T_initial = np.diag(Tii)
        for i in range(1, N):
            for j in range(i):
                T_initial[i, j] = 1 / dx * ((-1) ** (i - j)) / (i - j)
        T_final = T_initial - T_initial.T
        return T_final

    @staticmethod
    def second_derivative(grid):
        '''
        This function returns the matrix to dot into your eigen function to return the second derivative using a
        modification of eq A7 from DOI:10.1063/1.462100
        :param grid: The grid that the values are calculated from
        :type grid: np array (1 dimension)
        :return: A matrix that when dotted into the values of your function, will return the second derivative of that
                function
        :rtype: np array (2 dimensions)
        '''
        import numpy as np
        N = len(grid)
        a = grid[0]
        b = grid[-1]
        coeff = (1. / ((-1) / (((float(N)) / (b - a)) ** 2)))

        Tii = np.zeros(N)

        Tii += coeff * ((np.pi ** 2.) / 3.)
        T_initial = np.diag(Tii)
        for i in range(1, N):
            for j in range(i):
                T_initial[i, j] = coeff * ((-1.) ** (i - j)) * (2. / ((i - j) ** 2))
        T_final = T_initial + T_initial.T - np.diag(Tii)
        return T_final

    @staticmethod
    def first_derivative_fd(grid):
        '''
        This function returns a matrix that when dotted into the values from the function of interest, will return the
        first derivative of that function
        :param grid: The grid that the values are calculated from
        :type grid: np array (1 dimension)
        :return: A matrix that when dotted into the values of your function, will return the derivative of that function
        :rtype: np array (2 dimensions)
        '''
        import numpy as np, scipy.sparse as sp
        dx = (grid[-1] - grid[0]) / (len(grid))
        coeffs = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]) / dx

        fd_matrix = sp.diags(coeffs, np.arange(-2, 3, 1), shape=(len(grid), len(grid)))

        return fd_matrix.toarray()

    @staticmethod
    def second_derivative_fd(grid):
        '''
        This function returns a matrix that when dotted into the values from the function of interest, will return the
        second derivative of that function
        :param grid: The grid that the values are calculated from
        :type grid: np array (1 dimension)
        :return: A matrix that when dotted into the values of your function, will return the second derivative of that
                function
        :rtype: np array (2 dimensions)
        '''
        import numpy as np, scipy.sparse as sp
        dx = (grid[-1] - grid[0]) / (len(grid))
        coeffs = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]) / (dx ** 2)

        fd_matrix = sp.diags(coeffs, np.arange(-2, 3, 1), shape=(len(grid), len(grid)))

        return fd_matrix.toarray()

    @staticmethod
    def kron_sum(b, a):
        '''

        :param b: derivative matrix 2
        :type b: np array (2 dimensions)
        :param a: derivative matrix 1
        :type a: np array (2 dimensions)
        :return: a sparse matrix representation of the two derivative matrices of interest that can be dotted into the
        values of the function of interest
        :rtype: sp array
        '''
        import scipy.sparse as sp
        '''Computes a Kronecker sum to build our Kronecker-Delta tensor product expression'''
        n_1 = a.shape[0]  # len of grid 1
        n_2 = b.shape[0]  # len of grid 2
        ident_1 = sp.identity(n_1)  # the identity matrix of grid 1
        ident_2 = sp.identity(n_2)  # the identity matrix of grid 2

        return sp.kron(a, ident_2) + sp.kron(ident_1, b)

    def sparse_2d_fd_mat_1st(self, gridz):
        '''
        This function constructs a sparse matrix representation of the derivative matrix used to obtain the first
        derivative of the 2d eigen function of interest
        :param gridz: A list of the grids of interest to construct the first derivative matrix
        :type gridz: list (2 dimensions)
        :return: A sparse representation of the derivative matrix
        :rtype: sp array
        '''
        import scipy.sparse as sp
        if self.fd is None:
            der = [self.first_derivative(g) for g in gridz]
        else:
            der = [self.first_derivative_fd(g) for g in gridz]
        der_map = map(sp.csr_matrix, der)

        from functools import reduce
        d = reduce(self.kron_sum, der_map)
        return d

    def sparse_2d_fd_mat_2nd(self, gridz):
        '''
        This function constructs a sparse matrix representation of the derivative matrix used to obtain the second
        derivative of the 2d eigen function of interest
        :param gridz: A list of the grids of interest to construct the second derivative matrix
        :type gridz: list (2 dimensions)
        :return: A sparse representation of the derivative matrix
        :rtype: sp array
        '''
        import scipy.sparse as sp
        if self.fd is None:
            der = [self.second_derivative(g) for g in gridz]
        else:
            der = [self.second_derivative_fd(g) for g in gridz]
        der_map = map(sp.csr_matrix, der)

        from functools import reduce
        d = reduce(self.kron_sum, der_map)
        return d

    def derivatives_2d(self):
        '''
        This function calculates the derivatives of 2d functions on an evenly spaced grid depending on the derivative of
        interest
        :return: The derivative of a 2d function on an evenly spaced grid
        :rtype: np array
        '''
        import scipy.sparse as sp
        gridz1 = [self.grid1, self.grid2]
        gridz2 = [self.grid2, self.grid1]
        if self._dx == 1:
            if self._dy == 1:
                fd1 = self.sparse_2d_fd_mat_1st(gridz1)
                fd2 = self.sparse_2d_fd_mat_1st(gridz2)
                return sp.csr_matrix.dot(fd1, sp.csr_matrix.dot(fd2, self.vals))
            else:
                fd = self.sparse_2d_fd_mat_1st(gridz1)
                return sp.csr_matrix.dot(fd, self.vals)
        elif self._dy == 1:
            fd = self.sparse_2d_fd_mat_1st(gridz2)
            return sp.csr_matrix.dot(fd, self.vals)
        elif self._dx == 2:
            fd = self.sparse_2d_fd_mat_2nd(gridz1)
            return sp.csr_matrix.dot(fd, self.vals)
        elif self._dy ==2:
            fd = self.sparse_2d_fd_mat_2nd(gridz2)
            return sp.csr_matrix.dot(fd, self.vals)
        else:
            print("I can't do those types of derivatives yet")
            raise ValueError

    def compute_derivative(self, dx, dy=None):
        '''
        This function calculates the derivatives of either a 1d or a 2d eigen function on an evenly spaced grid of
        points
        :param dx: The derivative with respect to grid1
        :type dx: int (1 or 2)
        :param dy: The derivative with respect to grid2
        :type dy: int (1 or 2) or None
        :return: The derivative of the 1d or 2d function on the same evenly spaced grid
        :rtype: np array
        '''
        import numpy as np
        self._dx = dx
        self._dy = dy
        if self.grid2 is None:
            if self.fd is None:
                if self._dx == 1:
                    return np.dot(self.first_derivative(self.grid1), self.vals)
                elif self._dx == 2:
                    return np.dot(self.second_derivative(self.grid1), self.vals)
                else:
                    print("I can't do that derivative yet")
                    raise ValueError
            else:
                if self._dx == 1:
                    return np.dot(self.first_derivative_fd(self.grid1), self.vals)
                elif self._dx == 2:
                    return np.dot(self.second_derivative_fd(self.grid1), self.vals)
                else:
                    print("I can't do that derivative yet")
                    raise ValueError
        else:
            return self.derivatives_2d()