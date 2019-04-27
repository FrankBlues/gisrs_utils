# -*- coding: utf-8 -*-
"""Some interpolation method.

@author: Administrator
"""
import math

import numpy as np
from scipy.spatial import cKDTree as KDTree
# http://docs.scipy.org/doc/scipy/reference/spatial.html

import scipy.linalg.decomp_lu as lu_decomp


def bilinear_interpolation(x, y, points):
    """ 双线性插值方法
    https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
    Interpolate (x,y) from values associated with four points.

    Args:
        x, y (float): The x, y coordinate.
        points (list): The four points are a list of four triplets:
            (x, y, value). The four points can be in any order.
            They should form a rectangle.

    Raises:
        ValueError: If points do not form a rectangle or
                (x, y) not within the rectangle

    Examples:
        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    Todo:
        * Change to numpy ndarray.

    """
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1) + 0.0)


class Invdisttree:
    """反距离权重（IDW)插值方法
    inverse-distance-weighted interpolation using KDTree:
        invdisttree = Invdisttree( X, z )  -- data points, values
        interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        terpolates z from the 3 points nearest each query point q;

        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1/distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?

        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.

    """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ?
    # todo: regular grid, nnear ndim+1, 2*ndim
    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


class Krig2d(object):
    """ 改写IDL中克里格插值方法.
    KEYWORD PARAMETERS:
      Model Parameters:
        EXPONENTIAL: if set (with parameters [A, C0, C1]), use an exponential
               semivariogram model.
        SPHERICAL:   if set (with parameters [A, C0, C1]), use a spherical
               semivariogram model.

      Both models use the following parameters:
        A:    the range. At distances beyond A, the semivariogram
            or covariance remains essentialy constant.
            See the definition of the functions below.
        C0:   the "nugget," which provides a discontinuity at the
            origin.
        C1:   the covariance value for a zero distance, and the variance
            of the random sample Z variable. If only a two element
            vector is supplied, C1 is set to the sample variance.
            (C0 + C1) = the "sill," which is the variogram value for
            very large distances.

    Todo:
        * 未完成

    """
    def __init__(self, z, x, y, t, krig_method="exponential"):
        """Initial class.

        Args:
            z, x, y (numpy ndarray): Arrays containing the Z, X, and Y
                    coordinates of the data points on the surface.
                    For irregular grids, all three parameters must be
                    present and have the same number of elements.
            t: list [[A, C0] or [A , C0 , C1]] or [Range,Nugget,sill(Scale)]
            krig_method (str):  Models, (exponential,spherical,gaussian,linear)

        Raises:
            ValueError: If x, y, and z have different number of elements or
                        krig_method not in the given method.

        """
        self.z = z
        if z.size != x.size or z.size != y.size:
            raise ValueError("x, y, and z must have same number of elements.")
        self.x = x
        self.y = y

        if krig_method not in ["exponential", "spherical", "gaussian",
                               "linear"]:
            raise ValueError("Method must be one of exponential, spherical,"
                             "gaussian, linear.")

        # default value for C1
        if len(t) == 2:
            t.append(z.var() - t[1])

        self.t = t

    @staticmethod
    def _krig_expon(d, t):
        """ Return Exponential Covariance Fcn
             C(d) = C1 exp(-3 d/A)   if d > 0
                  = C1 + C0  if d == 0
        """
        r = t[2] * math.exp((-3./t[0]) * d)
        r[d == 0] = t[1] + t[2]
        return r

    @staticmethod
    def _krig_sphere(d, t):
        """ Return Spherical Covariance Fcn
             C(d) = C1 [ 1 - 1.5(d/A) + 0.5(d/A)^3] if d < A
                  = C1 + C0  if d == 0
                  = 0        if d > A
        """
        # d>A时，r = 1
        r = d/t[0] if d/t[0] < 1 else 1
        v = t[2]*(1 - r*(1.5 - 0.5*r*r))
        v[r == 0] = t[1] + t[2]
        return v

    @staticmethod
    def _krig_gaussian(d, t):
        """ Return Gaussian Covariance Fcn
             C(d) = C1 exp(-3 d^2/A^2)   if d > 0
                  = C1 + C0  if d == 0

        """
        r = t[2] * math.exp(-3.*(d/t[0])**2)
        r[d == 0] = t[1] + t[2]
        return r

    @staticmethod
    def _krig_linear(d, t):
        """ Return Gaussian Covariance Fcn
             C(d) = C1 (1 - d/A)   if d > 0
                  = C1 + C0  if d == 0

        """
        r = d/t[0] if d/t[0] < 1 else 1
        v = t[2] * (1 - r)
        v[d == 0] = t[1] + t[2]
        return v

    @staticmethod
    def _call_func(func, *args):
        return func(*args)

    def interp(self):

        if self.krig_method == "exponential":
            fname = self._krig_expon
        elif self.krig_method == "spherical":
            fname = self._krig_sphere
        elif self.krig_method == "gaussian":
            fname = self._krig_gaussian
        elif self.krig_method == "linear":
            fname = self._krig_linear
        else:
            raise ValueError("You must choose one of the following models: "
                             "EXPONENTIAL, GAUSSIAN, LINEAR, or SPHERICAL ")

        n = self.z.size

        # of eqns to solve
        m = n + 1
        a = np.zeros((m, m), dtype='float32')

        # Construct the symmetric distance matrix.
        for i in range(n - 1):
            j = np.arange(n - i) + i
            # Distance squared
            d = (self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2

            a[i, j] = d
            a[j, i] = d

        # Get coefficient matrix
        a = self._call_func(fname, np.sqrt(a), self.t)
        a[:, n] = 1.0            # Fill edges
        a[n, :] = 1.0
        a[n, n] = 0.0

        lu_and_piv = lu_decomp.lu_factor(a)

        # solve ax = b
        az = lu_decomp.lu_solve(lu_and_piv,np.append(self.z,0))


if __name__ == '__main__':
    pass
