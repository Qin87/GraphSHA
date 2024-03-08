import numpy as np
import torch
from numpy import linalg as LA
from scipy.sparse import coo_matrix

'''
def hermitian_decomp(A, q = 0.25):
    # this function is only tested based on the numpy array
    # should be updated if a sparse matrix is required

    A_upper = np.triu(A)
    A_lower = np.triu(A.T)

    #get A_s
    A_s = -np.ones_like(A)
    avg_mask = np.logical_and((A_upper > 0), (A_lower > 0))

    # u,v and v,u both exist position
    pos = (avg_mask == True)
    A_s[pos] = 0.5*(A_upper[pos] + A_lower[pos])

    # only one of u,v and v,u exists
    pos = (avg_mask == False)
    A_s[pos] = A_upper[pos] + A_lower[pos]
    A_s = np.triu(A_s) + np.triu(A_s).T

    # phase
    theta = 2*np.pi*q*((A_upper - A_lower) + (A_lower - A_upper).T)

    # degree
    D_s = np.diag(np.sum(A_s, axis = 1))

    # eigendecomposition
    L = D_s - A_s*np.exp(1j*theta)
    w, v = LA.eig(L) # column of v is the right eigenvector

    return L, w, v
'''


###########################################
####### Dense implementation ##############
###########################################
def cheb_poly(A, K):
    K += 1
    N = A.shape[0]  # [N, N]
    multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian[0] += np.eye(N, dtype=np.float32)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] += A
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] += 2 * np.dot(A, multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2]

    return multi_order_laplacian


def decomp(A, q, norm, laplacian, max_eigen, gcn_appr):
    A = 1.0 * np.array(A)
    if gcn_appr:
        A += 1.0 * np.eye(A.shape[0])

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.sum(np.array(A_sym), axis=0)
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = np.diag(d)
        A_sym = np.dot(np.dot(D, A_sym), D)

    if laplacian:
        Theta = 2 * np.pi * q * 1j * (A - A.T)  # phase angle array
        if norm:
            D = np.diag([1.0] * len(d))
        else:
            d = np.sum(np.array(A_sym), axis=0)  # diag of degree array
            D = np.diag(d)
        L = D - np.exp(Theta) * A_sym
    '''
    else:
        #transition matrix
        d_out = np.sum(np.array(A), axis = 1)
        d_out[d_out==0] = -1
        d_out = 1.0/d_out
        d_out[d_out<0] = 0
        D = np.diag(d_out)
        L = np.eye(len(d_out)) - np.dot(D, A)
    '''
    w, v = None, None
    if norm:
        if max_eigen == None:
            w, v = LA.eigh(L)
            L = (2.0 / np.amax(np.abs(w))) * L - np.diag([1.0] * len(A))
        else:
            L = (2.0 / max_eigen) * L - np.diag([1.0] * len(A))
            w = None
            v = None

    return L, w, v


def hermitian_decomp(As, q=0.25, norm=False, laplacian=True, max_eigen=None, gcn_appr=False):
    ls, ws, vs = [], [], []
    if len(As.shape) > 2:
        for i, A in enumerate(As):
            l, w, v = decomp(A, q, norm, laplacian, max_eigen, gcn_appr)
            vs.append(v)
            ws.append(w)
            ls.append(l)
    else:
        ls, ws, vs = decomp(As, q, norm, laplacian, max_eigen, gcn_appr)
    return np.array(ls), np.array(ws), np.array(vs)


###########################################
####### Sparse implementation #############
###########################################
def cheb_poly_sparse(A, K):
    '''
    T0(x) = 1, T1(x) = x, and Tk(x) = 2xTk−1(x) + Tk−2(x)
    Args:
        A:
        K:

    Returns: Chebyshev polynomial

    '''
    K += 1
    N = A.shape[0]  # [N, N]
    # multi_order_laplacian = np.zeros([K, N, N], dtype=np.complex64)  # [K, N, N]
    multi_order_laplacian = []
    multi_order_laplacian.append(coo_matrix((np.ones(N), (np.arange(N), np.arange(N))),shape=(N, N), dtype=np.float32))
    # creates a square sparse matrix with ones along the diagonal and zeros elsewhere.
    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian.append(A)
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian.append(2.0 * A.dot(multi_order_laplacian[k - 1]) - multi_order_laplacian[k - 2])  # from the third element
                #

    return multi_order_laplacian


def hermitian_decomp_sparse(row, col, size, q=0.25, norm=True, laplacian=True, max_eigen=2,
                            gcn_appr=False, edge_weight=None):
    '''

    Args:
        row: src node
        col: tgt node
        size: num of all nodes
        q:
        norm: normalization
        laplacian:
        max_eigen:
        gcn_appr: add self-loop
        edge_weight:

    Returns:signed directed magnetic Laplacian with the Hermitian adjacency matrix

    '''
    row = row.detach().cpu().numpy()        # use this, or row = row.detach().numpy() won't work in GPU
    col = col.detach().cpu().numpy()

    if edge_weight is None:
        A = coo_matrix((np.ones(len(row)), (row, col)), shape=(size, size), dtype=np.float32)   # creates a sparse matrix A where the non-zero elements are located at the coordinates specified by row and col, and each non-zero element has a value of 1.
    else:
        A = coo_matrix((edge_weight.detach().numpy(), (row, col)), shape=(size, size), dtype=np.float32)

    diag = coo_matrix((np.ones(size), (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)    #  creates a sparse diagonal matrix diag where all off-diagonal elements are zero, and each diagonal element has a value of 1.
    if gcn_appr:
        A += diag

    A_sym = 0.5 * (A + A.T)  # symmetrized adjacency

    if norm:
        d = np.array(A_sym.sum(axis=0))[0]  # out degree  # d will be a 1D NumPy array containing the out-degree of each node in the graph represented by the matrix A_sym
        d[d == 0] = 1
        d = np.power(d, -0.5)
        D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        A_sym = D.dot(A_sym).dot(D)

    if laplacian:
        Theta = 2 * np.pi * q * 1j * (A - A.T)  # phase angle array # Each element of Theta will represent the phase angle value associated with the difference between corresponding elements of A and its transpose.
        Theta.data = np.exp(Theta.data)     # computes the element-wise exponential of the phase angle values stored in the matrix Theta  # Accessing the .data attribute of a sparse matrix in SciPy returns the non-zero elements of the sparse matrix
        if norm:
            D = diag
        else:
            d = np.sum(A_sym, axis=0)  # diag of degree array
            D = coo_matrix((d, (np.arange(size), np.arange(size))), shape=(size, size), dtype=np.float32)
        L = D - Theta.multiply(A_sym)  # element-wise  # L= D − H= D− A.P,

    if norm:
        L = (2.0 / max_eigen) * L - diag

    return L