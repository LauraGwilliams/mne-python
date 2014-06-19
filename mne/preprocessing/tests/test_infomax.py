# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

"""
Test the infomax algorithm.
Parts of this code are taken from scikit-learn
"""

import numpy as np
from scipy import stats
from scipy import linalg

from nose.tools import assert_true

from numpy.testing import assert_almost_equal
from mne.preprocessing.infomax_ import infomax


def center_and_norm(x, axis=-1):
    """ Centers and norms x **in place**

        Parameters
        -----------
        x: ndarray
            Array with an axis of observations (statistical units) measured on
            random variables.
        axis: int, optional
            Axis along which the mean and variance are calculated.
    """
    x = np.rollaxis(x, axis)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)


def test_gs():
    """
    Test gram schmidt orthonormalization
    """
    from sklearn.decomposition.fastica_ import _gs_decorrelation

    # generate a random orthogonal  matrix
    rng = np.random.RandomState(0)
    W, _, _ = np.linalg.svd(rng.randn(10, 10))
    w = rng.randn(10)
    _gs_decorrelation(w, W, 10)
    assert_true((w ** 2).sum() < 1.e-10)
    w = rng.randn(10)
    u = _gs_decorrelation(w, W, 5)
    tmp = np.dot(u, W.T)
    assert_true((tmp[:5] ** 2).sum() < 1.e-10)


def test_fastica_simple(add_noise=False):
    """ Test the infomax algorithm on very simple data.
    """
    from sklearn.decomposition import RandomizedPCA
    rng = np.random.RandomState(0)
    # scipy.stats uses the global RNG:
    np.random.seed(0)
    n_samples = 1000
    # Generate two sources:
    s1 = (2 * np.sin(np.linspace(0, 100, n_samples)) > 0) - 1
    s2 = stats.t.rvs(1, size=n_samples)
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing angle
    phi = 0.6
    mixing = np.array([[np.cos(phi),  np.sin(phi)],
                       [np.sin(phi), -np.cos(phi)]])
    m = np.dot(mixing, s)

    if add_noise:
        m += 0.1 * rng.randn(2, 1000)

    center_and_norm(m)

    # function as fun arg
    def g_test(x):
        return x ** 3, (3 * x ** 2).mean(axis=-1)

    algos = [True, False]
    for algo in algos:
        X = RandomizedPCA(n_components=2, whiten=True).fit_transform(m.T)
        k_ = infomax(X, extended=algo)
        s_ = np.dot(X, k_)

        center_and_norm(s_)
        s1_, s2_ = s_
        # Check to see if the sources have been estimated
        # in the wrong order
        if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
            s2_, s1_ = s_
        s1_ *= np.sign(np.dot(s1_, s1))
        s2_ *= np.sign(np.dot(s2_, s2))

        # Check that we have estimated the original sources
        if not add_noise:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)
        else:
            assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=1)
            assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=1)


def test_non_square_infomax(add_noise=False):
    """ Test the infomax algorithm on very simple data.
    """
    from sklearn.decomposition import RandomizedPCA, fastica

    rng = np.random.RandomState(0)

    n_samples = 1000
    # Generate two sources:
    t = np.linspace(0, 100, n_samples)
    s1 = np.sin(t)
    s2 = np.ceil(np.sin(np.pi * t))
    s = np.c_[s1, s2].T
    center_and_norm(s)
    s1, s2 = s

    # Mixing matrix
    n_observed = 6
    mixing = rng.randn(n_observed, 2)
    m = np.dot(mixing, s)

    if add_noise:
        m += 0.1 * rng.randn(n_observed, n_samples)

    center_and_norm(m)
    pca = RandomizedPCA(n_components=2, whiten=True, random_state=rng)
    m = pca.fit_transform(m.T).T
    unmixing_ = infomax(m.T, random_state=rng)
    # XXX compare / debug
    # _, mixing_, s_ = fastica(m.T, random_state=rng, whiten=False)
    s_ = np.dot(m.T, unmixing_.T)
    s_ = s_.T

    # Check that the mixing model described in the docstring holds:
    assert_almost_equal(s_, np.dotun(mixing_, m))

    center_and_norm(s_)
    s1_, s2_ = s_
    # Check to see if the sources have been estimated
    # in the wrong order
    if abs(np.dot(s1_, s2)) > abs(np.dot(s1_, s1)):
        s2_, s1_ = s_
    s1_ *= np.sign(np.dot(s1_, s1))
    s2_ *= np.sign(np.dot(s2_, s2))

    # Check that we have estimated the original sources
    if not add_noise:
        assert_almost_equal(np.dot(s1_, s1) / n_samples, 1, decimal=2)
        assert_almost_equal(np.dot(s2_, s2) / n_samples, 1, decimal=2)
