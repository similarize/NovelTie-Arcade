# Purple Bear (she): soft SDF bear, seen only on the phone (video call from her home). Per-frame mouth
# (open J, smile) re-sculpted in a small box; head tilt/nod via a smooth head weight. Soles at z=0.
import numpy as np
from skimage import measure
from frog_sdf import ell, sph, cap, smin, ssub, ss
import mfrog as MF

HEAD_C = np.array([0, 0, 1.08]); HEAD_R = 0.30
MUZ_C = np.array([0, -0.255, 1.01])
EYES = [np.array([s * 0.105, -0.255, 1.13]) for s in (1, -1)]
X0, X1, Y0, Y1, Z0, Z1 = -0.42, 0.42, -0.40, 0.32, 0.0, 1.50


def body(x, y, z):
    d = ell(x, y, z, (0, 0.02, 0.52), (0.30, 0.26, 0.40))
    d = smin(d, sph(x, y, z, HEAD_C, HEAD_R), 0.10)
    for s in (1, -1):
        d = smin(d, sph(x, y, z, (s * 0.21, 0.03, 1.32), 0.085), 0.04)               # ears
        d = smin(d, cap(x, y, z, (s * 0.25, 0.0, 0.78), (s * 0.30, -0.10, 0.45), 0.085, 0.075), 0.05)   # arms
        d = smin(d, cap(x, y, z, (s * 0.14, 0.0, 0.25), (s * 0.15, -0.05, 0.06), 0.11, 0.10), 0.05)    # legs
    d = smin(d, ell(x, y, z, MUZ_C, (0.125, 0.09, 0.085)), 0.04)
    return d


def full(P, x, y, z):
    J = float(P.get('J', 0)); sm = float(P.get('smile', 0.6))
    d = body(x, y, z)
    for s in (1, -1):   # ear inner dips
        d = ssub(d, sph(x, y, z, (s * 0.21, -0.07, 1.33), 0.045), 0.02)
    # mouth: smile crease under the nose + an oval opening that grows with J (never a slot)
    mz = MUZ_C[2] - 0.035
    cx = x / 0.085
    zc = mz + (0.012 + 0.02 * sm) * cx ** 2
    crease = np.maximum(np.maximum(np.abs(z - zc) - 0.004, np.abs(x) - 0.085), y + 0.27)
    d = ssub(d, crease, 0.004)
    if J > 0.004:
        cav = ell(x, y, z, (0, MUZ_C[1] - 0.06, mz - 0.6 * J), (0.055 + 0.3 * J, 0.09, 0.9 * J + 0.004))
        d = ssub(d, cav, 0.008)
    return d


def build(P, vox=0.004):
    xs = np.arange(X0, X1, vox, dtype=np.float32); ys = np.arange(Y0, Y1, vox, dtype=np.float32); zs = np.arange(Z0, Z1, vox, dtype=np.float32)
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    for i, xv in enumerate(xs):
        vol[i] = full(P, np.full_like(Y, xv), Y, Z)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox, vox, vox))
    v += np.array([xs[0], ys[0], zs[0]], np.float32)
    v = MF.taubin(v, f)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    A = {}
    A['rest'] = v.copy()
    A['muzzle'] = (1 - ss(0.9, 1.05, np.sqrt((x / 0.13) ** 2 + ((y - MUZ_C[1]) / 0.11) ** 2 + ((z - MUZ_C[2]) / 0.095) ** 2))).astype(np.float32)
    A['belly'] = ((1 - ss(0.85, 1.0, np.sqrt((x / 0.19) ** 2 + ((z - 0.52) / 0.27) ** 2))) * ss(0.0, -0.08, y)).astype(np.float32)
    A['mouthin'] = (ss(0.004, 0.010, -body(x, y, z))).astype(np.float32)
    A['nose'] = (1 - ss(0.9, 1.1, np.sqrt((x / 0.05) ** 2 + ((y + 0.345) / 0.04) ** 2 + ((z - 1.035) / 0.032) ** 2))).astype(np.float32)
    # head nod/tilt: rigid head transform blended in by height
    hw = ss(0.78, 0.92, z)
    R = MF.rotm(*P.get('head', (0, 0, 0)))
    piv = np.array([0, 0, 0.85])
    vh = (v - piv) @ R.T + piv
    v = v * (1 - hw[:, None]) + vh * hw[:, None]
    Gh = np.eye(4); Gh[:3, :3] = R; Gh[:3, 3] = piv - R @ piv
    return v.astype(np.float32), f.astype(np.int32), A, Gh.astype(np.float32)
