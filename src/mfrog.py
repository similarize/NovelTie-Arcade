# Movie froggy: ONE shared parameterized frog for James / Jimmy / Bubbles / Rexy (identical shape; only colour +
# accessory differ, so they can never drift). Built on the pixar_test v2/v3 SDF head + mouth (frog_sdf3.Mouth:
# smile crease, crescent/oval domain-warp opening, true O, smirk, cheeks, tongue) plus legs, feet with toe pads
# and rest-pose arms with 4-finger frog hands. Pose = linear-blend skinning in numpy (fast, cached rest mesh);
# only the mouth box is re-sculpted when the mouth changes.
# Frog-local axes: x = frog's left, -y = front, z up. Ground (soles) at z = GROUND; output verts are shifted so
# the soles sit at z = 0 (object origin between the feet).
import numpy as np, os, time
from skimage import measure
import scipy.sparse as sp
import frog_sdf3 as F3
from frog_sdf import ell, sph, cap, smin, ssub, ss, EYE_C, EYE_R

GROUND = -0.365
X0, X1, Y0, Y1, Z0, Z1 = -0.40, 0.40, -0.37, 0.28, -0.385, 1.08
BOX = dict(x0=-0.34, x1=0.34, y0=-0.37, y1=0.07, z0=0.44, z1=0.92)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- rest skeleton (frog-local, pre-shift coords)
J = {}
for s, n in ((1, 'L'), (-1, 'R')):
    J['sh_' + n] = np.array([s * 0.168, 0.03, 0.548])
    J['el_' + n] = np.array([s * 0.290, 0.005, 0.365])
    J['wr_' + n] = np.array([s * 0.292, -0.060, 0.210])
    J['hip_' + n] = np.array([s * 0.100, 0.030, 0.200])
    J['kn_' + n] = np.array([s * 0.178, -0.060, -0.060])
    J['an_' + n] = np.array([s * 0.150, 0.030, -0.300])
J['hips'] = np.array([0, 0.02, 0.22]); J['chest'] = np.array([0, 0.02, 0.42]); J['head'] = np.array([0, 0.0, 0.60])
J['jet'] = np.array([0, 0.24, 0.50])
BONES = [  # name, parent, pivot
    ('hips', None, 'hips'), ('chest', 'hips', 'chest'), ('head', 'chest', 'head'), ('jet', 'chest', 'jet'),
    ('uarm_L', 'chest', 'sh_L'), ('farm_L', 'uarm_L', 'el_L'), ('hand_L', 'farm_L', 'wr_L'),
    ('uarm_R', 'chest', 'sh_R'), ('farm_R', 'uarm_R', 'el_R'), ('hand_R', 'farm_R', 'wr_R'),
    ('thigh_L', 'hips', 'hip_L'), ('shin_L', 'thigh_L', 'kn_L'), ('foot_L', 'shin_L', 'an_L'),
    ('thigh_R', 'hips', 'hip_R'), ('shin_R', 'thigh_R', 'kn_R'), ('foot_R', 'shin_R', 'an_R'),
]
BI = {b[0]: i for i, b in enumerate(BONES)}


def nrm(v):
    v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-9)


def hand_frame(s):
    a = nrm(J['wr_' + ('L' if s > 0 else 'R')] - J['el_' + ('L' if s > 0 else 'R')])   # along the hand (down)
    a = nrm(a + np.array([0, -0.10, -0.25]))
    t = nrm(np.array([0, -1.0, 0]) - a * a[1] * -1.0)                                 # thumb forward
    t = nrm(t - a * np.dot(t, a))
    n = nrm(np.cross(a, t)) * (-s if np.cross(a, t)[0] * s > 0 else 1)                 # palm faces the body
    if np.dot(n, [-s, 0, 0]) < 0: n = -n
    return np.stack([t, a, n], 1)


def limb_parts(s):
    n = 'L' if s > 0 else 'R'
    sh, el, wr = J['sh_' + n], J['el_' + n], J['wr_' + n]
    hip, kn, an = J['hip_' + n], J['kn_' + n], J['an_' + n]
    hand = F3.hand_parts(wr + nrm(wr - el) * 0.004, hand_frame(s), 0.22, 0.35, 0.15)
    foot_c = np.array([s * 0.165, -0.070, GROUND + 0.030])
    toes = []
    for k, ang in enumerate(np.radians([-34, -12, 10, 32])):
        d = np.array([s * np.sin(ang) * 1.0, -np.cos(ang), 0.0])
        b0 = foot_c + np.array([s * 0.010, -0.05, -0.004])
        tip = b0 + d * (0.105 - 0.012 * abs(k - 1.5))
        tip[2] = GROUND + 0.022
        toes.append((b0, tip))
    return dict(sh=sh, el=el, wr=wr, hand=hand, hip=hip, kn=kn, an=an, foot_c=foot_c, toes=toes)


LP = {s: limb_parts(s) for s in (1, -1)}


def arm_sdf(s, x, y, z):
    P = LP[s]
    d = cap(x, y, z, P['sh'], P['el'], 0.058, 0.047)
    mid = P['el'] + (P['wr'] - P['el']) * 0.38
    d = smin(d, cap(x, y, z, P['el'], mid, 0.046, 0.044), 0.018)
    d = smin(d, cap(x, y, z, mid, P['wr'], 0.044, 0.036), 0.012)
    C = {'hand': (None, (P['wr'], hand_frame(s), 0.22, 0.35, 0.15))}
    h = F3.hand_sdf(C, x, y, z, P['hand'])
    return smin(d, h, 0.014)


def leg_sdf(s, x, y, z, pads_only=False):
    P = LP[s]
    pad = np.full_like(x, 10.0)
    for b0, tip in P['toes']:
        pad = np.minimum(pad, sph(x, y, z, tip + np.array([0, -0.004, 0.0]), 0.021))
    if pads_only: return pad
    d = cap(x, y, z, P['hip'], P['kn'], 0.092, 0.058)
    d = smin(d, ell(x, y, z, (P['hip'] + P['kn']) / 2 + np.array([0, 0.012, 0]), (0.088, 0.080, 0.15)), 0.04)  # meaty thigh
    d = smin(d, cap(x, y, z, P['kn'], P['an'], 0.056, 0.036), 0.03)
    d = smin(d, ell(x, y, z, P['foot_c'], (0.060, 0.085, 0.026)), 0.035)
    d = smin(d, cap(x, y, z, P['an'], P['foot_c'] + np.array([0, 0.02, 0.0]), 0.036, 0.030), 0.02)
    for b0, tip in P['toes']:
        t = cap(x, y, z, b0, tip, 0.017, 0.012)
        d = smin(d, smin(t, sph(x, y, z, tip + np.array([0, -0.004, 0.0]), 0.021), 0.006), 0.012)
    return d


def parts_sdf(x, y, z):
    """dict of part distances (rest pose)."""
    body = F3.outer_body(x, y, z)
    return dict(body=body, arm_L=arm_sdf(1, x, y, z), arm_R=arm_sdf(-1, x, y, z),
                leg_L=leg_sdf(1, x, y, z), leg_R=leg_sdf(-1, x, y, z))


def combine(head_d, x, y, z):
    """body (or mouth-sculpted body) + limbs; arms blend only at the shoulder, legs at the hip."""
    d = head_d
    for s in (1, -1):
        a = arm_sdf(s, x, y, z)
        n = 'L' if s > 0 else 'R'
        near = np.exp(-np.sum([(c - J['sh_' + n][i]) ** 2 for i, c in enumerate((x, y, z))], 0) / 0.09 ** 2)
        d = smin(d, a, 0.004 + 0.045 * near)
        l = leg_sdf(s, x, y, z)
        nearh = np.exp(-np.sum([(c - J['hip_' + n][i]) ** 2 for i, c in enumerate((x, y, z))], 0) / 0.14 ** 2)
        d = smin(d, l, 0.004 + 0.06 * nearh)
    return d


def static_fn(x, y, z):
    return combine(F3.static_sdf(x, y, z), x, y, z)


def grid(vox):
    return (np.arange(X0, X1, vox, dtype=np.float32), np.arange(Y0, Y1, vox, dtype=np.float32),
            np.arange(Z0, Z1, vox, dtype=np.float32))


def eval_vol(fn, xs, ys, zs, dtype=np.float32):
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys.astype(dtype), zs.astype(dtype), indexing='ij')
    for i, x in enumerate(xs):
        vol[i] = fn(np.full_like(Y, x), Y, Z)
    return vol


_SV = {}
def static_volume(vox, cache_dir=None):
    key = int(round(vox * 1e4))
    if key in _SV: return _SV[key]
    cache_dir = cache_dir or os.environ.get('FROG_CACHE', os.path.join(HERE, '..', 'cache'))
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f'mfrog_static_{key}.npy')
    xs, ys, zs = grid(vox)
    if os.path.exists(p):
        v = np.load(p)
        if v.shape == (len(xs), len(ys), len(zs)):
            _SV[key] = v; return v
    v = eval_vol(static_fn, xs, ys, zs)
    np.save(p, v); _SV[key] = v
    return v


def taubin(v, f, it=6):
    nv = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(nv, nv)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    for _ in range(it):
        for lam in (0.5, -0.53):
            v = v + lam * ((A @ v) / deg[:, None] - v)
    return v


MKEYS = ['smile', 'smirk', 'J', 'U', 'wide', 'pucker', 'fv', 'mbp']


QK = dict(smile=0.05, smirk=0.05, J=0.003, U=0.002, wide=0.05, pucker=0.05, fv=0.1, mbp=0.1)


def quant(P):
    Q = dict(P)
    for k, q in QK.items():
        Q[k] = round(float(P.get(k, 0.5 if k == 'smile' else 0.0)) / q) * q
    for k in ('brow_L', 'brow_R', 'echeek_L', 'echeek_R'):
        Q[k] = round(float(P.get(k, 0.0)) / 0.1) * 0.1
    return Q


def mouth_key(P):
    Q = quant(P)
    return tuple(round(Q[k], 4) for k in MKEYS)


def build_rest(P, vox):
    """Rest-pose mesh for mouth params P. Returns verts (pre-shift), faces, attrs dict, rest coords."""
    M = F3.Mouth(P)
    xs, ys, zs = grid(vox)
    vol = static_volume(vox).copy()
    ix0, ix1 = np.searchsorted(xs, BOX['x0']), np.searchsorted(xs, BOX['x1'])
    iy0, iy1 = np.searchsorted(ys, BOX['y0']), np.searchsorted(ys, BOX['y1'])
    iz0, iz1 = np.searchsorted(zs, BOX['z0']), np.searchsorted(zs, BOX['z1'])
    fn = lambda x, y, z: combine(M.full(x, y, z), x, y, z)
    vol[ix0:ix1, iy0:iy1, iz0:iz1] = eval_vol(fn, xs[ix0:ix1], ys[iy0:iy1], zs[iz0:iz1], np.float64)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(vox, vox, vox))
    v += np.array([X0, Y0, Z0], np.float32)
    v = taubin(v, f)
    A, dv, rest = attrs(M, v, P)
    return (v + dv).astype(np.float32), f.astype(np.int32), A, rest


def attrs(M, v, P):
    x, y, z = [v[:, i].astype(np.float64) for i in range(3)]
    inbox = (x > BOX['x0'] + 0.01) & (x < BOX['x1'] - 0.01) & (y < BOX['y1'] - 0.01) & (z > BOX['z0'] + 0.01) & (z < BOX['z1'] - 0.01)
    A = {}
    xr, yr, zr = x.copy(), y.copy(), z.copy()
    sr = np.full_like(x, 1.0); depth = np.zeros_like(x)
    if inbox.any():
        dcl, (a, b, c, s_) = M.closed(x[inbox], y[inbox], z[inbox])
        xr[inbox], yr[inbox], zr[inbox], sr[inbox] = a, b, c, s_
        depth[inbox] = -dcl
    A['mouthin'] = (ss(0.0045, 0.011, depth)).astype(np.float32)
    A['mouthdeep'] = (ss(0.014, 0.06, depth)).astype(np.float32)
    tg = np.zeros_like(x)
    if M.J > 0.008 and inbox.any():
        tg[inbox] = (1 - ss(-0.003, 0.004, M.tongue(x[inbox], y[inbox], z[inbox]))) * ss(0.002, 0.008, depth[inbox])
    A['tongue'] = tg.astype(np.float32)
    lat = 1 - ss(0.85, 1.02, np.abs(xr) / M.W0)
    A['lipline'] = (np.exp(-(sr / 0.0075) ** 2) * lat * (1 - ss(0.002, 0.008, depth)) * inbox).astype(np.float32)
    # cream throat + belly (v2 recipe), only on the body part
    PD = parts_sdf(xr, yr, zr)
    lim = np.minimum(np.minimum(PD['arm_L'], PD['arm_R']), np.minimum(PD['leg_L'], PD['leg_R']))
    bodyw = ss(-0.012, 0.012, lim - PD['body'])
    eh = np.sqrt((xr / 0.235) ** 2 + ((yr + 0.02) / 0.27) ** 2)
    zt = F3.MZ - 0.036 + 0.022 * (xr / 0.22) ** 2
    head_c = ss(0.0, 0.010, zt - zr) * (1 - ss(0.90, 0.97, eh)) * ss(0.50, 0.60, zr)
    be = np.sqrt((xr / 0.150) ** 2 + ((zr - 0.37) / 0.255) ** 2)
    belly_c = (1 - ss(0.935, 1.0, be)) * (1 - ss(-0.05, 0.03, yr))
    A['cream'] = (np.clip(head_c + belly_c - head_c * belly_c, 0, 1) * bodyw).astype(np.float32)
    pads = np.minimum(leg_sdf(1, xr, yr, zr, True), leg_sdf(-1, xr, yr, zr, True))
    A['pad'] = (1 - ss(-0.002, 0.004, pads)).astype(np.float32)
    # space-suit masks: white below the neck, grey gloves + boots
    A['suitw'] = (1 - ss(0.585, 0.610, zr)).astype(np.float32)
    glove = ss(0.03, 0.0, np.minimum(PD['arm_L'], PD['arm_R'])) * ss(0.25, 0.20, zr) * ss(0.24, 0.27, np.abs(xr))
    boot = ss(-0.20, -0.24, zr)
    A['suitg'] = np.clip(np.maximum(glove * (1 - bodyw), boot), 0, 1).astype(np.float32)
    # ---- skin weights (soft part assignment, then split along each chain)
    T = 0.006
    names = ['body', 'arm_L', 'arm_R', 'leg_L', 'leg_R']
    D = np.stack([PD[k] for k in names], 1)
    D = D - D.min(1, keepdims=True)
    Wp = np.exp(-D / T); Wp /= Wp.sum(1, keepdims=True)
    W = np.zeros((len(x), len(BONES)), np.float64)
    head_tot = ss(0.50, 0.65, zr); chest = (1 - head_tot) * ss(0.26, 0.44, zr); hips = 1 - head_tot - chest
    W[:, BI['head']] += Wp[:, 0] * head_tot; W[:, BI['chest']] += Wp[:, 0] * chest; W[:, BI['hips']] += Wp[:, 0] * hips
    Q = np.stack([xr, yr, zr], 1)
    def seg_t(a, b):
        ab = b - a; return np.clip(((Q - a) @ ab) / ab.dot(ab), -0.5, 1.5)
    for s, n, k in ((1, 'L', 1), (-1, 'R', 2)):
        t1 = seg_t(J['sh_' + n], J['el_' + n]); t2 = seg_t(J['el_' + n], J['wr_' + n])
        fore = ss(0.80, 1.12, t1)
        hand = ss(0.88, 1.06, t2) * fore
        up = 1 - fore; fo = fore - hand
        # shoulder: the top of the upper arm blends to the chest
        shb = ss(-0.05, 0.25, t1)
        W[:, BI['uarm_' + n]] += Wp[:, k] * up * shb; W[:, BI['chest']] += Wp[:, k] * up * (1 - shb)
        W[:, BI['farm_' + n]] += Wp[:, k] * fo; W[:, BI['hand_' + n]] += Wp[:, k] * hand
        kk = k + 2
        t1 = seg_t(J['hip_' + n], J['kn_' + n]); t2 = seg_t(J['kn_' + n], J['an_' + n])
        shin = ss(0.78, 1.10, t1)
        foot = ss(0.86, 1.08, t2) * shin
        hb = ss(-0.10, 0.30, t1)
        W[:, BI['thigh_' + n]] += Wp[:, kk] * (1 - shin) * hb; W[:, BI['hips']] += Wp[:, kk] * (1 - shin) * (1 - hb)
        W[:, BI['shin_' + n]] += Wp[:, kk] * (shin - foot); W[:, BI['foot_' + n]] += Wp[:, kk] * foot
    W /= W.sum(1, keepdims=True)
    A['W'] = W.astype(np.float32)
    # brow + eye-cheek bulges (v2), applied in rest space
    dv = np.zeros((len(x), 3))
    for sgn, nm in ((1, 'L'), (-1, 'R')):
        r = np.sqrt((xr - sgn * EYE_C[0]) ** 2 + (yr - EYE_C[1] - 0.02) ** 2 + (zr - EYE_C[2] - 0.09) ** 2)
        re = np.sqrt((xr - sgn * EYE_C[0]) ** 2 + (yr - EYE_C[1]) ** 2 + (zr - EYE_C[2]) ** 2)
        w = np.exp(-(r / 0.07) ** 2) * ss(EYE_R + 0.002, EYE_R + 0.012, re) * float(P.get('brow_' + nm, 0))
        dv[:, 1] += -0.005 * w; dv[:, 2] += 0.02 * w
        rc = np.sqrt((xr - sgn * EYE_C[0]) ** 2 + (yr - EYE_C[1] + 0.06) ** 2 + (zr - EYE_C[2] + 0.105) ** 2)
        wc = np.exp(-(rc / 0.06) ** 2) * ss(EYE_R + 0.002, EYE_R + 0.012, re) * float(P.get('echeek_' + nm, 0))
        dv[:, 0] += sgn * 0.003 * wc; dv[:, 1] += -0.006 * wc; dv[:, 2] += 0.012 * wc
    rest = np.stack([xr, yr, zr], 1).astype(np.float32)
    return A, dv.astype(np.float32), rest


# ---------------------------------------------------------------- pose / FK / skinning
def rotm(rx=0, ry=0, rz=0):
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def T4(R=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R is not None: M[:3, :3] = R
    M[:3, 3] = t; return M


def fk(pose):
    """pose: dict bone -> [rx,ry,rz] deg (in rest/world axes, about the bone pivot); 'sq' = hips squash (z scale).
    Returns dict bone -> 4x4 skinning matrix (rest -> posed, frog-local, pre-shift coords)."""
    G = {}
    for name, par, piv in BONES:
        p = J[piv]
        R = _R(pose.get(name, (0, 0, 0)))
        L = T4(R, p) @ T4(None, -p)
        if name == 'hips':
            s = float(pose.get('sq', 1.0)); S = np.diag([1 / np.sqrt(s), 1 / np.sqrt(s), s, 1.0])
            fp = np.array([0, 0.02, 0.10])     # squash about the pelvis floor
            L = T4(None, fp) @ S @ T4(None, -fp) @ L
        if name == 'head':
            s = float(pose.get('hsq', 1.0)); S = np.diag([1 / np.sqrt(s), 1 / np.sqrt(s), s, 1.0])
            L = T4(None, p) @ S @ T4(None, -p) @ L
        G[name] = (G[par] @ L) if par else L
    # legs counter the hips rotation/squash when planted: handled by the animator (foot IK) upstream
    return G


def skin(v, W, G):
    Ms = np.stack([G[b[0]] for b in BONES])          # (B,4,4)
    vh = np.concatenate([v, np.ones((len(v), 1), np.float32)], 1)
    out = np.zeros((len(v), 3), np.float64)
    for i in range(len(BONES)):
        w = W[:, i]
        m = w > 1e-4
        if not m.any(): continue
        out[m] += w[m, None] * (vh[m] @ Ms[i].T)[:, :3]
    return out.astype(np.float32)


def _R(v):
    v = np.asarray(v, float)
    return v if v.shape == (3, 3) else rotm(*v)


def leg_ik(pose, foot_targets, foot_rot=None):
    """Plant feet. foot_targets = {'L': ankle position (frog-local, pre-shift)}; knee bends forward.
    Numeric 2-bone solve (thigh rx/rz, shin rx), then the foot is counter-rotated to foot_rot (default flat)."""
    from scipy.optimize import minimize
    for n, tgt in foot_targets.items():
        tgt = np.asarray(tgt, float)
        def err(q):
            p = dict(pose); p['thigh_' + n] = [q[0], 0, q[1]]; p['shin_' + n] = [q[2], 0, 0]
            G = fk(p)
            a = (G['shin_' + n] @ np.append(J['an_' + n], 1))[:3]
            return np.sum((a - tgt) ** 2) * 1e4 + 1e-6 * (q[2] - 0) ** 2 + (max(0, -q[2]) ** 2) * 1e-2
        q0 = [-10.0, 0.0, 20.0]
        r = minimize(err, q0, method='Nelder-Mead', options=dict(xatol=0.05, fatol=1e-6, maxiter=400))
        q = r.x
        pose['thigh_' + n] = [q[0], 0, q[1]]; pose['shin_' + n] = [q[2], 0, 0]
        G = fk(pose)
        par = G['shin_' + n][:3, :3]
        want = _R(foot_rot) if foot_rot is not None else np.eye(3)
        pose['foot_' + n] = par.T @ want
    return pose
