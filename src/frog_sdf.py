# Jimmy body mesh from a smooth signed-distance sculpt -> marching cubes -> Taubin smooth.
# Blender axes: x right, -y = front (faces camera), z up. Units: metres, frog ~1 m tall.
# Output: frog_body.npz (verts, faces) consumed by build_scene.py inside Blender.
import numpy as np
from skimage import measure

VOX = 0.0033
X0, X1 = -0.37, 0.37
Y0, Y1 = -0.31, 0.26
Z0, Z1 = 0.10, 1.07

# --- shared mouth constants (build_scene.py mirrors these) ---
LIP_Z0, LIP_CURVE, LIP_HALFW = 0.700, 0.050, 0.282
SLIT_W = 0.0055
SLIT_CY, SLIT_RY = 0.065, 0.32
EYE_C = (0.165, -0.05, 0.952); EYE_R = 0.098

def zl(x):
    return LIP_Z0 + LIP_CURVE * (x / LIP_HALFW) ** 2

def ell(px, py, pz, c, r):
    qx, qy, qz = (px - c[0]) / r[0], (py - c[1]) / r[1], (pz - c[2]) / r[2]
    k0 = np.sqrt(qx * qx + qy * qy + qz * qz)
    k1 = np.sqrt((qx / r[0]) ** 2 + (qy / r[1]) ** 2 + (qz / r[2]) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

def sph(px, py, pz, c, r):
    return np.sqrt((px - c[0]) ** 2 + (py - c[1]) ** 2 + (pz - c[2]) ** 2) - r

def cap(px, py, pz, a, b, ra, rb):
    a = np.array(a); b = np.array(b); ba = b - a
    pax, pay, paz = px - a[0], py - a[1], pz - a[2]
    h = np.clip((pax * ba[0] + pay * ba[1] + paz * ba[2]) / ba.dot(ba), 0, 1)
    dx, dy, dz = pax - ba[0] * h, pay - ba[1] * h, paz - ba[2] * h
    return np.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * h)

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

def ssub(a, b, k):  # a minus b
    h = np.clip(0.5 - 0.5 * (a + b) / k, 0, 1)
    return a * (1 - h) + (-b) * h + k * h * (1 - h)

def outer(px, py, pz):
    """Skin without the mouth carve (also used for the colour masks)."""
    d = ell(px, py, pz, (0, 0.0, 0.765), (0.315, 0.250, 0.175))
    d = smin(d, ell(px, py, pz, (0, 0.04, 0.810), (0.250, 0.200, 0.140)), 0.05)
    d = smin(d, ell(px, py, pz, (0, -0.010, 0.665), (0.288, 0.228, 0.115)), 0.06)
    for s in (-1, 1):  # eye bumps
        d = smin(d, sph(px, py, pz, (s * EYE_C[0], 0.0, 0.935), 0.106), 0.045)
    # torso + belly
    d = smin(d, ell(px, py, pz, (0, 0.035, 0.42), (0.215, 0.170, 0.285)), 0.12)
    d = smin(d, ell(px, py, pz, (0, -0.005, 0.38), (0.190, 0.150, 0.220)), 0.05)
    # arms
    for s in (-1, 1):
        d = smin(d, cap(px, py, pz, (s * 0.165, 0.03, 0.555), (s * 0.268, 0.0, 0.365), 0.058, 0.046), 0.045)
        d = smin(d, cap(px, py, pz, (s * 0.268, 0.0, 0.365), (s * 0.250, -0.09, 0.215), 0.046, 0.040), 0.02)
    return d

def full(px, py, pz):
    d = outer(px, py, pz)
    # nostrils
    for s in (-1, 1):
        d = ssub(d, sph(px, py, pz, (s * 0.048, -0.247, 0.815), 0.0085), 0.007)
    # eye sockets (eyeball objects sit in them)
    for s in (-1, 1):
        d = ssub(d, sph(px, py, pz, (s * EYE_C[0], EYE_C[1], EYE_C[2]), EYE_R + 0.0045), 0.008)
    # mouth cavity
    cav = ell(px, py, pz, (0, -0.045, 0.690), (0.212, 0.160, 0.074))
    # slit: slab around lip line inside an ellipse in the XY plane
    e = np.sqrt((px / LIP_HALFW) ** 2 + ((py - SLIT_CY) / SLIT_RY) ** 2) - 1.0
    slab = np.abs(pz - zl(px)) - SLIT_W
    slit = np.maximum(slab, e * 0.25)
    slit = np.maximum(slit, py - 0.10)
    d = ssub(d, np.minimum(cav, slit), 0.0045)
    return d

def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def rig_data(v):
    x, y, z = v[:, 0].astype(np.float64), v[:, 1].astype(np.float64), v[:, 2].astype(np.float64)
    o = outer(x, y, z)
    d = z - zl(x)
    ax = np.abs(x) / LIP_HALFW
    sx = np.sign(x)
    er = np.sqrt((x / LIP_HALFW) ** 2 + ((y - SLIT_CY) / SLIT_RY) ** 2)
    A = {}
    # colour masks
    mouthin = ss(0.004, 0.010, -o) * (1 - ss(0.78, 0.82, z))
    eh = np.sqrt((x / 0.255) ** 2 + ((y + 0.03) / 0.27) ** 2)
    head_c = ss(0.002, 0.008, -d) * (1 - ss(0.92, 1.0, eh)) * ss(0.52, 0.58, z)
    be = np.sqrt((x / 0.150) ** 2 + ((z - 0.40) / 0.285) ** 2)
    belly_c = (1 - ss(0.94, 1.0, be)) * (1 - ss(-0.03, 0.03, y))
    A['cream'] = np.maximum(head_c, belly_c).astype(np.float32)
    A['mouthin'] = mouthin.astype(np.float32)
    # skin weights
    wsplit = 0.003 + 0.035 * ss(0.95, 1.2, er)
    jaw_split = ss(-wsplit, wsplit, -d)
    head_tot = ss(0.50, 0.65, z)
    jaw = jaw_split * (1 - ss(0.04, 0.16, y)) * ss(0.52, 0.63, z)
    jaw = np.minimum(jaw, head_tot)
    head = np.clip(head_tot - jaw, 0, 1)
    chest = (1 - head_tot) * ss(0.26, 0.44, z)
    hips = (1 - head_tot) * (1 - ss(0.26, 0.44, z))
    for k, w in dict(w_jaw=jaw, w_head=head, w_chest=chest, w_hips=hips).items():
        A[k] = w.astype(np.float32)
    # shape keys (deltas in rest space)
    inslit = 1 - ss(0.97, 1.08, er)
    f = 1 - ss(SLIT_W + 0.0015, SLIT_W + 0.022, np.abs(d))
    seal = -np.sign(d) * np.minimum(np.abs(d), SLIT_W) * f * inslit
    m = 1 - ss(0.35, 0.65, ax)
    Z = np.zeros_like(x)
    def key(dx, dy, dz):
        return np.stack([dx + Z, dy + Z, dz + Z], 1).astype(np.float32)
    A['sk_lipseal_mid'] = key(0, 0, seal * m)
    A['sk_lipseal_side'] = key(0, 0, seal * (1 - m))
    g = np.exp(-(d / 0.05) ** 2) * (1 - ss(1.0, 1.35, er))
    front = 1 - ss(-0.02, 0.06, y)
    smile_dz = 0.036 * ax ** 2 * g
    smile_dy = 0.014 * ax ** 3 * g
    cdx = Z.copy(); cdy = Z.copy(); cdz = Z.copy()
    for s in (-1, 1):
        c = np.exp(-(((x - s * 0.21) ** 2 + (z - 0.775) ** 2 + (y + 0.12) ** 2) / 0.065 ** 2)) * (o > -0.003)
        cdx += s * 0.5 * 0.011 * c; cdy += -0.5 * 0.011 * c; cdz += 0.6 * 0.011 * c
    A['sk_smile'] = key(cdx, smile_dy + cdy, smile_dz + cdz)
    # sides of the lower lip drop with the jaw (wider 'D' opening); upper-lip sides lift a touch
    lo = (d < 0) * np.exp(-((d + 0.01) / 0.04) ** 2) * inslit * ss(0.15, 0.6, ax) * (1 - ss(0.85, 1.0, ax))
    up = (d > 0) * np.exp(-((d - 0.01) / 0.03) ** 2) * inslit * ss(0.2, 0.7, ax) * (1 - ss(0.88, 1.0, ax))
    A['sk_open_sides'] = key(0, 0, -0.014 * lo + 0.006 * up)
    A['lipline'] = (np.exp(-(d / 0.011) ** 2) * (1 - ss(0.98, 1.06, er)) * (o > -0.004)).astype(np.float32)
    gu = (d < 0) * np.exp(-((d + 0.008) / 0.03) ** 2) * np.exp(-(x / 0.16) ** 2) * inslit
    A['sk_grin_u'] = key(0, 0, -0.022 * gu)
    A['sk_wide'] = key(sx * 0.016 * ax ** 2 * g, 0.008 * ax ** 2 * g, -0.003 * ax ** 2 * g)
    pc = np.exp(-(x / 0.11) ** 2) * np.exp(-(d / 0.035) ** 2) * front
    A['sk_pucker'] = key(-sx * 0.035 * ax ** 2 * g, -0.03 * pc, 0.006 * np.sign(d) * pc)
    pl = np.exp(-(x / 0.16) ** 2) * np.exp(-((d + 0.012) / 0.018) ** 2) * (d < 0.002) * front
    A['sk_fv'] = key(0, 0.010 * pl, 0.010 * pl)
    for s, nm in ((1, 'L'), (-1, 'R')):  # L = +x
        r = np.sqrt((x - s * EYE_C[0]) ** 2 + (y - EYE_C[1] - 0.02) ** 2 + (z - EYE_C[2] - 0.09) ** 2)
        re = np.sqrt((x - s * EYE_C[0]) ** 2 + (y - EYE_C[1]) ** 2 + (z - EYE_C[2]) ** 2)
        w = np.exp(-(r / 0.07) ** 2) * ss(EYE_R + 0.002, EYE_R + 0.012, re)
        A['sk_brow_' + nm] = key(0, -0.005 * w, 0.02 * w)
        rc = np.sqrt((x - s * EYE_C[0]) ** 2 + (y - EYE_C[1] + 0.06) ** 2 + (z - EYE_C[2] + 0.105) ** 2)
        wc = np.exp(-(rc / 0.06) ** 2) * ss(EYE_R + 0.002, EYE_R + 0.012, re)
        A['sk_cheek_' + nm] = key(s * 0.003 * wc, -0.006 * wc, 0.012 * wc)
    return A

if __name__ == '__main__':
    xs = np.arange(X0, X1, VOX, dtype=np.float32)
    ys = np.arange(Y0, Y1, VOX, dtype=np.float32)
    zs = np.arange(Z0, Z1, VOX, dtype=np.float32)
    print('grid', len(xs), len(ys), len(zs))
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    for i, x in enumerate(xs):
        vol[i] = full(np.full_like(Y, x), Y, Z)
    v, f, n, _ = measure.marching_cubes(vol, 0.0, spacing=(VOX, VOX, VOX))
    v += np.array([X0, Y0, Z0], np.float32)
    # skimage winding is already outward for this SDF sign convention
    print('mc', len(v), len(f))
    # Taubin smoothing (volume preserving) to remove voxel ripple
    nv = len(v)
    import scipy.sparse as sp
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    for it in range(6):
        for lam in (0.5, -0.53):
            avg = (A @ v) / deg[:, None]
            v = v + lam * (avg - v)
    attrs = rig_data(v)
    np.savez_compressed('frog_body.npz', verts=v.astype(np.float32), faces=f.astype(np.int32), **attrs)
    print('saved', list(attrs))
