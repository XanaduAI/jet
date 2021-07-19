from thewalrus import hafnian
from scipy.special import factorial as fac
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
from random import random
from thewalrus.quantum import pure_state_amplitude
import numpy as np
import time
import sys
import os

def complex_to_real_displacements(mu, hbar=2):
    r"""Returns the vector of complex displacements and conjugate displacements.
    Args:
        mu (array): length-:math:`2N` means vector
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the expectation values
        :math:`[\langle a_1\rangle, \langle a_2\rangle,\dots,\langle a_N\rangle, \langle a^\dagger_1\rangle, \dots, \langle a^\dagger_N\rangle]`
    """
    N = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:N] + 1j * mu[N:]) / np.sqrt(2 * hbar)
    # the expectation values (<a_1>, <a_2>,...,<a_N>, <a^\dagger_1>, ..., <a^\dagger_N>)
    return np.concatenate([alpha, alpha.conj()])

def reduction(A, rpt):
    r"""Calculates the reduction of an array by a vector of indices.
    This is equivalent to repeating the ith row/column of :math:`A`, :math:`rpt_i` times.
    Args:
        A (array): matrix of size [N, N]
        rpt (Sequence): sequence of N positive integers indicating the corresponding rows/columns
            of A to be repeated.
    Returns:
        array: the reduction of A by the index vector rpt
    """
    rows = [i for sublist in [[idx] * j for idx, j in enumerate(rpt)] for i in sublist]

    if A.ndim == 1:
        return A[rows]

    return A[:, rows][rows]


def Xmat(N):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`
    Args:
        N (int): positive integer
    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])
    return X


def Qmat(cov, hbar=2):
    r"""Returns the :math:`Q` Husimi matrix of the Gaussian state.
    Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the :math:`Q` matrix.
    """
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)

    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)
    return Q


def Amat(cov, hbar=2, cov_is_qmat=False):
    r"""Returns the :math:`A` matrix of the Gaussian state whose hafnian gives the photon number probabilities.
    Args:
        cov (array): :math:`2N\times 2N` covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cov_is_qmat (bool): if ``True``, it is assumed that ``cov`` is in fact the Q matrix.
    Returns:
        array: the :math:`A` matrix.
    """
    # number of modes
    N = len(cov) // 2
    X = Xmat(N)

    # inverse Q matrix
    if cov_is_qmat:
        Q = cov
    else:
        Q = Qmat(cov, hbar=hbar)

    Qinv = np.linalg.inv(Q)

    # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
    A = X @ (np.identity(2 * N) - Qinv).conj()
    return A


def pure_state_amplitude_local(mu, cov, i, include_prefactor=True, tol=1e-10, hbar=2, check_purity=False, quad=False, recursive=False):
    r"""Returns the :math:`\langle i | \psi\rangle` element of the state ket
    of a Gaussian state defined by covariance matrix cov.
    Args:
        mu (array): length-:math:`2N` quadrature displacement vector
        cov (array): length-:math:`2N` covariance matrix
        i (list): list of amplitude elements
        include_prefactor (bool): if ``True``, the prefactor is automatically calculated
            used to scale the result.
        tol (float): tolerance for determining if displacement is negligible
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        check_purity (bool): if ``True``, the purity of the Gaussian state is checked
            before calculating the state vector.
    Returns:
        complex: the pure state amplitude
    """
    if check_purity:
        if not is_pure_cov(cov, hbar=hbar, rtol=1e-05, atol=1e-08):
            raise ValueError("The covariance matrix does not correspond to a pure state")

    rpt = i
    beta = complex_to_real_displacements(mu, hbar=hbar)
    Q = Qmat(cov, hbar=hbar)
    A = Amat(cov, hbar=hbar)
    (n, _) = cov.shape
    N = n // 2
    B = A[0:N, 0:N].conj()
    alpha = beta[0:N]

    if np.linalg.norm(alpha) < tol:
        # no displacement
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            B_rpt = reduction(B, rpt)
            haf = hafnian(B_rpt, quad=quad, recursive=recursive)
        else:
            haf = hafnian_repeated(B, rpt)
    else:
        gamma = alpha - B @ np.conj(alpha)
        if np.prod([k + 1 for k in rpt]) ** (1 / len(rpt)) < 3:
            B_rpt = reduction(B, rpt)
            np.fill_diagonal(B_rpt, reduction(gamma, rpt))
            haf = hafnian(B_rpt, loop=True, quad=quad, recursive=recursive)
        else:
            haf = hafnian_repeated(B, rpt, mu=gamma, loop=True)

    if include_prefactor:
        pref = np.exp(-0.5 * (np.linalg.norm(alpha) ** 2 - alpha.conj() @ B @ alpha.conj()))
        haf *= pref

    return haf / np.sqrt(np.prod(fac(rpt)) * np.sqrt(np.linalg.det(Q)))

if len(sys.argv) < 3:
    print("Please provide a file to load and option for random value.")
    print("Run script as python walrus.py <random=0/1>")
    exit(1)

filename = sys.argv[1]

if not os.path.isfile(filename):
    print(f"File {filename} is not valid.")
    exit(1)

prog = sf.load(filename)
eng = sf.Engine("gaussian")
result = eng.run(prog)
cov = result.state.cov()
lw = 8
total_l = int(sys.argv[2])

amp = np.zeros(lw * lw).astype(int)

def random_sample_n_balls_m_bins(n, m):
    sample = np.zeros([m])
    bins = list(range(m))
    for i in range(n):
        ball = np.random.choice(bins)
        while sample[ball] == 3:
            ball = np.random.choice(bins)
        sample[ball] += 1
    sample = sample.astype(int)
    return sample

amp = random_sample_n_balls_m_bins(total_l, lw*lw)
mu = np.zeros(len(cov))
print("amp=",amp)
start = time.time()
ans = pure_state_amplitude_local(mu,cov,amp,check_purity=False)
end = time.time()

print("t=", end-start)
print("result=", ans)
