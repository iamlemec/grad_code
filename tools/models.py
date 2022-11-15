import jax
import jax.numpy as np

# compile `eig` on CPU to make it work
cpu, *_ = jax.devices('cpu')
eig = jax.jit(np.linalg.eig, device=cpu)

# general axis normalizer
def normed(A, axis=None):
    return A/np.sum(A, axis=axis, keepdims=True)

# this is just a thin wrapper around SVD (copied from scipy)
def null_space(A, rcond=None):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q
