import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import pi, sqrt, pow

# KM15 parameters
nval = 1.35
pval = 1.
nsea = 1.5
rsea = 1.
psea = 2.
bsea = 4.6
Mval = 0.789
rval = 0.918
bval = 0.4
C0 = 2.768
Msub = 1.204
Mtval = 3.993
rtval = 0.881
btval = 0.4
ntval = 0.6
Msea = sqrt(0.482)
rpi = 2.646
Mpi = 4.

def compute_km15_cffs(QQ, xB, t, k = 0.0):
    """
    ## Description:
    Evaluate the KM15 model for CFFs at the given kinematics.
    
    ## Returns:
        ReH, ImH, ReE, ReHt, ImHt, ReEt
    """

    # (X): Calculate the skewnesss, xi:
    xi = xB / (2.0 - xB)
    
    alpha_val = 0.43 + 0.85 * t
    alpha_sea = 1.13 + 0.15 * t
    Ct = C0 / (1.0 - t / Msub**2)**2

    def fHval(x):
        return (nval * rval / (1 + x) *
                ((2 * x) / (1 + x))**(-alpha_val) *
                ((1 - x) / (1 + x))**bval /
                (1 - ((1 - x) / (1 + x)) * (t / Mval**2))**pval)

    def fHsea(x):
        return (nsea * rsea / (1 + x) *
                ((2 * x) / (1 + x))**(-alpha_sea) *
                ((1 - x) / (1 + x))**bsea /
                (1 - ((1 - x) / (1 + x)) * (t / Msea**2))**psea)

    def fHtval(x):
        return (ntval * rtval / (1 + x) *
                ((2 * x) / (1 + x))**(-alpha_val) *
                ((1 - x) / (1 + x))**btval /
                (1 - ((1 - x) / (1 + x)) * (t / Mtval**2)))

    def fImH(x):
        return pi * ((8. / 9.) * fHval(x) + (1. / 9.) * fHsea(x))

    def fImHt(x):
        return pi * (8. / 9.) * fHtval(x)

    def fPV_ReH(x):
        return -2. * x / (x + xi) * fImH(x)

    def fPV_ReHt(x):
        return -2. * xi / (x + xi) * fImHt(x)

    DR_ReH, _ = quad(fPV_ReH, 1e-6, 1.0, weight = 'cauchy', wvar = xi)
    DR_ReHt, _ = quad(fPV_ReHt, 1e-6, 1.0, weight = 'cauchy', wvar = xi)

    # (X): Re[H]:
    real_h_km15 = DR_ReH / pi - Ct
    
    # (X): Im[H]:
    imag_h_km15 = fImH(xi)

    # (X): Re[E]:
    real_e_km15 = Ct

    # (X): Re[Ht]:
    real_ht_km15 = DR_ReHt / pi

    # (X): Im[Ht]:
    imag_ht_km15 = fImHt(xi)

    # (X): Re[Et]:
    real_et_km15 = rpi / xi * 2.164 / ((0.0196 - t) * (1.0 - t / Mpi**2)**2)

    return real_h_km15, imag_h_km15, real_e_km15, real_ht_km15, imag_ht_km15, real_et_km15