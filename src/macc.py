# Accessory shells that hug the shared frog body (rest pose), meshed once and skinned with the body bones:
# bandana (neck band + front triangle + side knot with two tails) and jetpack straps (+ chest strap).
# Also the space-suit collar/helmet are rigid (Blender side).
import numpy as np
from skimage import measure
import frog_sdf3 as F3
import mfrog as MF
from frog_sdf import ell, sph, cap, smin, ssub, ss


def _mesh(fn, lo, hi, vox):
    xs = np.arange(lo[0], hi[0], vox); ys = np.arange(lo[1], hi[1], vox); zs = np.arange(lo[2], hi[2], vox)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    for i, x in enumerate(xs):
        vol[i] = fn(np.full_like(Y, x), Y, Z)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox, vox, vox))
    v += np.array([xs[0], ys[0], zs[0]], np.float32)
    v = MF.taubin(v, f, 4)
    return v.astype(np.float32), f.astype(np.int32)


def body_d(x, y, z):
    return F3.outer_body(x, y, z)


def bandana_fn(x, y, z):
    d = body_d(x, y, z)
    shell = np.abs(d - 0.011) - 0.0045
    # neck band follows the chin junction: a little higher at the back
    zb = 0.585 + 0.03 * ss(-0.1, 0.15, y)
    band = np.abs(z - zb) - 0.030
    # front triangle (apex down at the belly), slightly offset to the frog's right like a worn bandana
    apex = 0.405
    w = 0.215 * np.clip((z - apex) / (zb - apex), 0, 1.2)
    tri = np.maximum(np.abs(x + 0.01) - w, np.maximum(apex - z, z - zb))
    tri = np.maximum(tri, y + 0.02)
    region = np.minimum(band, tri)
    b = np.maximum(shell, region)
    # rolled top edge
    b = smin(b, np.maximum(np.abs(d - 0.012) - 0.007, np.abs(z - (zb + 0.026)) - 0.008), 0.004)
    # knot on the frog's left side + two tails
    kc = np.array([0.205, -0.135, 0.585])
    b = smin(b, ell(x, y, z, kc, (0.034, 0.030, 0.030)), 0.01)
    t1 = ell(x, y, z, kc + np.array([0.060, -0.010, 0.030]), (0.062, 0.010, 0.026))
    t2 = ell(x, y, z, kc + np.array([0.050, -0.025, -0.040]), (0.050, 0.010, 0.030))
    b = smin(b, np.minimum(t1, t2), 0.012)
    return b


def straps_fn(x, y, z):
    d = body_d(x, y, z)
    shell = np.abs(d - 0.008) - 0.0035
    st = np.full_like(x, 10.0)
    for s in (-1, 1):
        st = np.minimum(st, np.abs(x - s * 0.118) - 0.019)
    st = np.maximum(st, np.maximum(0.30 - z, z - 0.60))
    chest = np.maximum(np.abs(z - 0.43) - 0.012, np.abs(x) - 0.14)
    chest = np.maximum(chest, y + 0.05)
    return np.maximum(shell, np.minimum(st, chest))


def weights(v):
    """simple body weights (head/chest/hips by height) for the accessory shells."""
    z = v[:, 2].astype(np.float64)
    W = np.zeros((len(v), len(MF.BONES)), np.float32)
    head = ss(0.56, 0.66, z) * 0.6
    chest = (1 - head) * ss(0.26, 0.44, z); hips = 1 - head - chest
    W[:, MF.BI['head']] = head; W[:, MF.BI['chest']] = chest; W[:, MF.BI['hips']] = hips
    return W


def build(vox=0.004):
    out = {}
    v, f = _mesh(bandana_fn, (-0.33, -0.33, 0.36), (0.36, 0.30, 0.68), vox)
    out['bandana'] = (v, f, weights(v))
    v, f = _mesh(straps_fn, (-0.25, -0.30, 0.26), (0.25, 0.30, 0.66), vox)
    out['straps'] = (v, f, weights(v))
    return out


if __name__ == '__main__':
    import time; t = time.time(); o = build(); print({k: len(v[0]) for k, v in o.items()}, round(time.time() - t, 1))
