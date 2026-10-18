# v3 Jimmy body: v2's per-frame SDF mouth + articulated arms with readable 4-finger frog hands.
# Arms are no longer baked into the static body: each frame poses an IK arm (shoulder -> elbow -> wrist)
# and a parametric hand (palm, 4 fingers with toe pads, thumb; curl/spread/thumb channels), smooth-unioned
# into the body only around the shoulder (so a hand on the chest touches it instead of melting into it).
# Mouth: true rounded "O" for o/u/w (upper lip lifts with a circular profile, corners pinch in), stronger smirk
# and cheek push. Throat/cheek jiggle from a spring channel. Per-vertex 'velocity' for arm motion blur.
#   python3 frog_sdf3.py --list 1,2,3 [--out meshes3] [--anim anim3.json]
# Blender axes: x right (screen right = Jimmy's left), -y = front, z up. Metres.
import numpy as np, json, os, sys, time
from skimage import measure
import scipy.sparse as sp
from frog_sdf import ell, sph, cap, smin, ssub, ss, EYE_C, EYE_R

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = float(os.environ.get('FROG_VOX', 0.0025))
X0, X1 = -0.52, 0.52
Y0, Y1 = -0.40, 0.26
Z0, Z1 = 0.10, 1.07
BOX = dict(x0=-0.37, x1=0.37, z0=0.40, z1=0.95, y1=0.22)   # per-frame head/mouth region
FPS = 24

MZ = 0.692
W_BASE = 0.232
CW = 0.0032
SHOULDER = {'R': np.array([-0.168, 0.03, 0.548]), 'L': np.array([0.168, 0.03, 0.548])}
UPPER, FORE = 0.205, 0.180


def outer_body(px, py, pz):
    """frog_sdf.outer without the arms."""
    d = ell(px, py, pz, (0, 0.0, 0.765), (0.315, 0.250, 0.175))
    d = smin(d, ell(px, py, pz, (0, 0.04, 0.810), (0.250, 0.200, 0.140)), 0.05)
    d = smin(d, ell(px, py, pz, (0, -0.010, 0.665), (0.288, 0.228, 0.115)), 0.06)
    for s in (-1, 1):
        d = smin(d, sph(px, py, pz, (s * EYE_C[0], 0.0, 0.935), 0.106), 0.045)
    d = smin(d, ell(px, py, pz, (0, 0.035, 0.42), (0.215, 0.170, 0.285)), 0.12)
    d = smin(d, ell(px, py, pz, (0, -0.005, 0.38), (0.190, 0.150, 0.220)), 0.05)
    # shoulder pads (where the arms plug in), so the torso has shoulders even with the arms posed away
    for s in (-1, 1):
        d = smin(d, sph(px, py, pz, (s * 0.170, 0.03, 0.545), 0.050), 0.05)
    return d


def static_sdf(px, py, pz):
    d = outer_body(px, py, pz)
    for s in (-1, 1):
        d = ssub(d, sph(px, py, pz, (s * 0.048, -0.247, 0.815), 0.0085), 0.007)
        d = ssub(d, sph(px, py, pz, (s * EYE_C[0], EYE_C[1], EYE_C[2]), EYE_R + 0.0045), 0.008)
    return d


# ------------------------------------------------------------------ arms + hands
def nrm(v):
    v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-9)


def rot(axis, ang):
    a = nrm(axis); c, s = np.cos(ang), np.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def frame_from(d, up):
    """orthonormal frame with local y along d; returns 3x3 (columns x,y,z)."""
    y = nrm(d); x = np.cross(y, up)
    if np.linalg.norm(x) < 1e-6: x = np.cross(y, [0, 0, 1.0])
    x = nrm(x); z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def arm_pose(P, S):
    """Skeleton for side S ('R' = Jimmy's right = -x). Returns dict of components:
       name -> (kind, params, R (3x3), origin) ; R/origin define the component's rigid frame."""
    sg = -1.0 if S == 'R' else 1.0
    sh = SHOULDER[S]
    W = np.array(P['w' + S], float)
    pole = np.array(P.get('pole' + S, [sg * 0.6, 0.5, -0.4]), float)
    v = W - sh; L = np.linalg.norm(v)
    L = min(L, (UPPER + FORE) * 0.995); vd = nrm(v); W = sh + vd * L
    # elbow via law of cosines, bent toward the pole
    ca = (UPPER ** 2 + L ** 2 - FORE ** 2) / (2 * UPPER * L)
    ang = np.arccos(np.clip(ca, -1, 1))
    pp = nrm(pole - vd * pole.dot(vd))
    E = sh + UPPER * (np.cos(ang) * vd + np.sin(ang) * pp)
    a = nrm(P['ha' + S]); n = np.array(P['hn' + S], float); n = nrm(n - a * n.dot(a))
    t = -sg * np.cross(a, n)            # thumb side
    Rh = np.stack([t, a, n], 1)         # hand frame: x = thumb side, y = along fingers, z = palm normal
    C = {}
    C['upper'] = ('limb', (sh, E, 0.050, 0.039), frame_from(E - sh, pp), sh)
    C['fore'] = ('limb', (E, W, 0.038, 0.031), frame_from(W - E, pp), E)
    C['hand'] = ('hand', (W, Rh, float(P.get('cu' + S, .3)), float(P.get('sp' + S, .3)), float(P.get('th' + S, .3))), Rh, W)
    return C, (sh, E, W)


def hand_parts(W, Rh, curl, spread, thumb):
    """List of primitives in world space: ('ell', c, R, r) / ('cap', a, b, ra, rb) / ('sph', c, r)."""
    t, a, n = Rh[:, 0], Rh[:, 1], Rh[:, 2]
    parts = []
    pc = W + a * 0.036 - n * 0.002
    parts.append(('ell', pc, Rh, (0.040, 0.036, 0.0170)))
    fan = np.radians([27, 9, -9, -27]) * (0.45 + 0.85 * spread)
    offs = [0.021, 0.007, -0.007, -0.021]
    lens = [(0.036, 0.029), (0.040, 0.032), (0.038, 0.030), (0.032, 0.026)]
    c1, c2 = np.radians(95) * curl, np.radians(100) * curl
    for fa, of, (l1, l2) in zip(fan, offs, lens):
        f = np.cos(fa) * a + np.sin(fa) * t
        b0 = W + a * 0.058 + t * of
        d1 = np.cos(c1) * f + np.sin(c1) * n
        j1 = b0 + d1 * l1
        d2 = np.cos(c1 + c2) * f + np.sin(c1 + c2) * n
        tip = j1 + d2 * l2
        parts.append(('cap', b0 - d1 * 0.008, j1, 0.0092, 0.0084))
        parts.append(('cap', j1, tip, 0.0084, 0.0078))
        parts.append(('sph', tip + d2 * 0.001 + n * 0.0015, 0.0125))     # frog toe pad
    # thumb: extended (thumb=0) points along the thumb side; curled (1) lies across the palm/fist
    tb = W + a * 0.020 + t * 0.028 + n * 0.006
    dt0 = nrm(0.30 * a + 1.0 * t - 0.05 * n)
    dt1 = nrm(0.75 * a - 0.15 * t + 0.70 * n)
    dt = nrm((1 - thumb) * dt0 + thumb * dt1)
    j = tb + dt * 0.030
    dt2 = nrm(dt + thumb * 0.6 * nrm(n - t))
    tip = j + dt2 * 0.026
    parts.append(('cap', tb, j, 0.0125, 0.0112))
    parts.append(('cap', j, tip, 0.0112, 0.0100))
    parts.append(('sph', tip + dt2 * 0.002, 0.0138))
    return parts


def prim_sdf(p, x, y, z):
    if p[0] == 'ell':
        _, c, R, r = p
        dx, dy, dz = x - c[0], y - c[1], z - c[2]
        qx = dx * R[0, 0] + dy * R[1, 0] + dz * R[2, 0]
        qy = dx * R[0, 1] + dy * R[1, 1] + dz * R[2, 1]
        qz = dx * R[0, 2] + dy * R[1, 2] + dz * R[2, 2]
        return ell(qx, qy, qz, (0, 0, 0), r)
    if p[0] == 'cap':
        return cap(x, y, z, p[1], p[2], p[3], p[4])
    return sph(x, y, z, p[1], p[2])


def limb_sdf(C, x, y, z):
    sh, E, r0, r1 = C['upper'][1]
    _, W, r2, r3 = C['fore'][1]
    d = cap(x, y, z, sh, E, r0, r1)
    mid = E + (W - E) * 0.38
    d = smin(d, cap(x, y, z, E, mid, r2, 0.041), 0.018)       # forearm swell
    d = smin(d, cap(x, y, z, mid, W, 0.041, r3), 0.012)
    return d


def hand_sdf(C, x, y, z, parts=None):
    W, Rh, cu, sp_, th = C['hand'][1]
    parts = parts or hand_parts(W, Rh, cu, sp_, th)
    d = prim_sdf(parts[0], x, y, z)
    # fingers: each finger chain fused tightly, then onto the palm with a small blend (no webbing at the tips)
    i = 1
    while i < len(parts):
        f = prim_sdf(parts[i], x, y, z)
        f = smin(f, prim_sdf(parts[i + 1], x, y, z), 0.004)
        f = smin(f, prim_sdf(parts[i + 2], x, y, z), 0.003)
        d = smin(d, f, 0.008)
        i += 3
    return d


def arm_bbox(C, pad=0.06):
    sh, E, W = C['upper'][1][0], C['upper'][1][1], C['fore'][1][1]
    pts = np.stack([sh, E, W, W + C['hand'][2][:, 1] * 0.13])
    return pts.min(0) - pad, pts.max(0) + pad


# ------------------------------------------------------------------ mouth (v2, with a rounder O and a stronger smirk)
class Mouth:
    def __init__(self, P):
        g = lambda k, dv=0.0: float(P.get(k, dv))
        self.sm = g('smile', 0.5); self.smirk = g('smirk'); self.J = g('J'); self.U = g('U')
        self.wide = g('wide'); self.puck = g('pucker'); self.fv = g('fv'); self.mbp = g('mbp')
        self.jig = g('jig')
        self.W0 = W_BASE * (1 + 0.07 * self.wide - 0.14 * self.puck + 0.03 * self.sm)
        self.Wo = self.W0 * (0.96 - 0.60 * self.puck + 0.02 * self.wide)    # O: opening pinches to the middle
        self.el = 0.72 - 0.30 * self.wide - 0.24 * self.puck
        self.eu = 1.25 - 0.75 * self.puck - 0.2 * self.wide
        self.gap0 = 0.0

    def zc(self, x):
        u = x / self.W0; au = np.abs(u)
        z = MZ + (0.018 + 0.030 * self.sm) * u * u * (1 - 0.5 * self.puck)
        z = z + (0.010 + 0.014 * self.sm) * np.clip((au - 0.72) / 0.28, 0, 1.3) ** 2
        z = z + self.smirk * (0.034 * np.clip(u, 0, 1.3) ** 2 - 0.007 * np.clip(-u, 0, 1.3) ** 2)
        return z

    def warp(self, x, y, z):
        s0 = z - self.zc(x)
        front = 1 - ss(-0.16, -0.04, y)
        lipz = np.exp(-(s0 / 0.045) ** 2) * np.clip(1 - (x / (self.W0 * 1.25)) ** 2, 0, 1)
        xr = x * (1 + 0.20 * self.puck * lipz * front)
        yr = y + 0.026 * self.puck * np.exp(-(s0 / 0.030) ** 2) * np.clip(1 - (x / (self.W0 * 0.8)) ** 2, 0, 1) * front
        yr = yr - 0.013 * self.fv * np.exp(-((s0 + 0.010) / 0.014) ** 2) * (1 - ss(-0.002, 0.006, s0)) \
            * np.clip(1 - (x / (self.W0 * 0.8)) ** 2, 0, 1) * front
        zc = self.zc(xr); s = z - zc
        u = np.abs(xr) / (self.Wo * 0.97)
        base = np.clip(1 - u * u, 0, 1)
        base_l = np.clip(1 - u ** (2.4 - 0.4 * self.puck), 0, 1)
        wy = 1 - ss(-0.17, 0.10, yr)
        sm_side = 1 + self.smirk * 0.35 * np.clip(xr / self.Wo, -1, 1)
        Dl = (self.J * base_l ** self.el * sm_side + self.gap0 * base ** 0.35) * wy
        Du = (self.U * base ** self.eu * sm_side + self.gap0 * base ** 0.35) * wy
        cwe = CW * (1 - ss(0.96, 1.15, np.abs(xr) / self.W0)) * (1 - ss(-0.02, 0.08, yr))
        Lu, Ll = 0.10, 0.17
        up = s + (cwe - Du) * (1 - ss(0, Lu, s - Du))
        lo = s + (Dl - cwe) * (1 - ss(0, Ll, -Dl - s))
        mid = -cwe + (s + Dl) / np.maximum(Du + Dl, 1e-9) * 2 * cwe
        sr = np.where(s >= Du, up, np.where(s <= -Dl, lo, mid))
        return xr, yr, zc + sr, sr

    def cheeks(self, x, y, z):
        d = np.full_like(x, 10.0)
        for sgn in (-1, 1):
            amt = self.sm * (1 + 0.9 * self.smirk * sgn) + 0.15 + 0.35 * self.smirk * (sgn > 0)
            cx = sgn * (self.W0 * 0.86); cz = float(self.zc(np.array(cx))) + 0.040 + 0.012 * self.sm
            cy = -0.25 * np.sqrt(max(1 - (cx / 0.315) ** 2 - ((cz - 0.765) / 0.175) ** 2, 0.05))
            n = np.array([cx / 0.315 ** 2, cy / 0.25 ** 2, (cz - 0.765) / 0.175 ** 2]); n /= np.linalg.norm(n)
            r = np.array([0.064, 0.052, 0.042])
            bul = 0.002 + 0.022 * amt + 0.004 * self.jig
            c = np.array([cx, cy, cz]) - n * (0.044 - bul)
            d = np.minimum(d, ell(x, y, z, c, r))
        return d

    def closed(self, x, y, z):
        xr, yr, zr, sr = self.warp(x, y, z)
        d = outer_body(xr, yr, zr)
        d = smin(d, self.cheeks(xr, yr, zr), 0.035)
        lat = 1 - ss(0.80, 1.02, np.abs(xr) / self.W0)
        roll = (0.0026 + 0.0018 * self.mbp + 0.0022 * self.puck) * lat * \
            (np.exp(-((sr - 0.011) / 0.0075) ** 2) + 1.1 * np.exp(-((sr + 0.012) / 0.0085) ** 2))
        d = d - roll * np.exp(-(d / 0.010) ** 2)
        for s in (-1, 1):
            d = ssub(d, sph(xr, yr, zr, (s * 0.048, -0.247, 0.815), 0.0085), 0.007)
            d = ssub(d, sph(xr, yr, zr, (s * EYE_C[0], EYE_C[1], EYE_C[2]), EYE_R + 0.0045), 0.008)
        return d, (xr, yr, zr, sr)

    def crease(self, xr, yr, zr, sr):
        ax = np.abs(xr)
        yb = 0.03 - 0.16 * (ax / self.W0) ** 2
        c = np.maximum(np.maximum(np.abs(sr) - CW, ax - self.W0), yr - yb)
        for sgn in (-1, 1):
            cx = sgn * self.W0
            cz = float(self.zc(np.array(cx)))
            cy = -0.25 * np.sqrt(max(1 - (cx / 0.315) ** 2 - ((cz - 0.765) / 0.175) ** 2, 0.05))
            c = np.minimum(c, sph(xr, yr, zr, (cx, cy + 0.004, cz), 0.0065 + 0.0015 * self.smirk * (sgn > 0)))
        return c

    def tongue(self, x, y, z):
        tz = MZ - 0.42 * self.J - 0.026
        r = (0.105 * (self.Wo / W_BASE) ** 0.5, 0.12, 0.026)
        return ell(x, y, z, (0, -0.075, tz), r)

    def full(self, x, y, z):
        d, (xr, yr, zr, sr) = self.closed(x, y, z)
        d = ssub(d, self.crease(xr, yr, zr, sr), 0.0042)
        if self.J > 0.008:
            d = smin(d, self.tongue(x, y, z), 0.010)
        return d


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


def grid():
    xs = np.arange(X0, X1, VOX, dtype=np.float32)
    ys = np.arange(Y0, Y1, VOX, dtype=np.float32)
    zs = np.arange(Z0, Z1, VOX, dtype=np.float32)
    return xs, ys, zs


def static_volume():
    path = os.path.join(HERE, f'static_vol3_{int(round(VOX * 1e4))}.npy')
    xs, ys, zs = grid()
    if os.path.exists(path):
        v = np.load(path)
        if v.shape == (len(xs), len(ys), len(zs)): return v
    vol = np.empty((len(xs), len(ys), len(zs)), np.float32)
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    for i, x in enumerate(xs):
        vol[i] = static_sdf(np.full_like(Y, x), Y, Z)
    np.save(path, vol)
    return vol


def eval_box(fn, xs, ys, zs, lo, hi):
    ix0, ix1 = np.searchsorted(xs, lo[0]), np.searchsorted(xs, hi[0])
    iy0, iy1 = np.searchsorted(ys, lo[1]), np.searchsorted(ys, hi[1])
    iz0, iz1 = np.searchsorted(zs, lo[2]), np.searchsorted(zs, hi[2])
    Xg, Yg, Zg = np.meshgrid(xs[ix0:ix1].astype(np.float64), ys[iy0:iy1].astype(np.float64), zs[iz0:iz1].astype(np.float64), indexing='ij')
    return (slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1)), fn(Xg, Yg, Zg), (Xg, Yg, Zg)


def to_local(C, name, pts):
    R, o = C[name][2], C[name][3]
    return (pts - o) @ R


def from_local(C, name, loc):
    R, o = C[name][2], C[name][3]
    return loc @ R.T + o


TEX_OFF = {'upper': np.array([3.0, 0, 0]), 'fore': np.array([5.0, 0, 0]), 'hand': np.array([7.0, 0, 0])}


def rig_attrs(M, v, P, ARM, Pn):
    x, y, z = [v[:, i].astype(np.float64) for i in range(3)]
    dcl, (xr, yr, zr, sr) = M.closed(x, y, z)
    # arm membership: which verts belong to an arm (and which component)
    armm = np.zeros(len(v)); comp = np.full(len(v), -1); side = np.full(len(v), -1)
    dbest = np.full(len(v), 1e3)
    names = ['upper', 'fore', 'hand']
    for si, S in enumerate(('R', 'L')):
        C = ARM[S][0]
        lo, hi = arm_bbox(C, 0.03)
        sel = np.where(np.all((v >= lo) & (v <= hi), 1))[0]
        if not len(sel): continue
        xs_, ys_, zs_ = x[sel], y[sel], z[sel]
        dl = limb_sdf(C, xs_, ys_, zs_); dh = hand_sdf(C, xs_, ys_, zs_)
        du = cap(xs_, ys_, zs_, *C['upper'][1])
        da = np.minimum(dl, dh)
        m = 1 - ss(-0.006, 0.012, da - dcl[sel])        # 1 where the arm is the nearer surface
        # fade the membership back into the body around the shoulder joint
        r_sh = np.linalg.norm(v[sel] - SHOULDER[S], axis=1)
        m = m * ss(0.035, 0.10, r_sh)
        better = da < dbest[sel]
        idx = sel[better]
        armm[idx] = np.maximum(armm[idx], m[better]); dbest[idx] = da[better]; side[idx] = si
        cc = np.where(dh <= dl, 2, np.where(du <= dl + 1e-9, 0, 1))
        comp[idx] = cc[better]
    A = {}
    depth = -dcl
    nb = 1 - armm
    A['mouthin'] = (ss(0.0045, 0.011, depth) * nb).astype(np.float32)
    A['mouthdeep'] = (ss(0.014, 0.06, depth) * nb).astype(np.float32)
    A['tongue'] = ((1 - ss(-0.003, 0.004, M.tongue(x, y, z))) * (M.J > 0.008) * ss(0.002, 0.008, depth) * nb).astype(np.float32)
    lat = 1 - ss(0.85, 1.02, np.abs(xr) / M.W0)
    A['lipline'] = (np.exp(-(sr / 0.0075) ** 2) * lat * (1 - ss(0.002, 0.008, depth)) * nb).astype(np.float32)
    eh = np.sqrt((xr / 0.235) ** 2 + ((yr + 0.02) / 0.27) ** 2)
    zt = MZ - 0.036 + 0.022 * (xr / 0.22) ** 2
    head_c = ss(0.0, 0.010, zt - zr) * (1 - ss(0.90, 0.97, eh)) * ss(0.50, 0.60, zr)
    be = np.sqrt((xr / 0.155) ** 2 + ((zr - 0.38) / 0.25) ** 2)
    belly_c = (1 - ss(0.935, 1.0, be)) * (1 - ss(-0.05, 0.03, yr))
    A['cream'] = (np.clip(head_c + belly_c - head_c * belly_c, 0, 1) * (1 - ss(0.2, 0.6, armm))).astype(np.float32)
    head_tot = ss(0.50, 0.65, zr)
    chest = (1 - head_tot) * ss(0.26, 0.44, zr)
    hips = (1 - head_tot) * (1 - ss(0.26, 0.44, zr))
    A['w_head'] = (head_tot * nb).astype(np.float32)
    A['w_chest'] = (chest * nb + armm).astype(np.float32)
    A['w_hips'] = (hips * nb).astype(np.float32)
    A['armmask'] = armm.astype(np.float32)
    # brow, eye-cheek bulges + throat/cheek jiggle, applied directly (body only)
    dv = np.zeros((len(x), 3))
    for sgn, nm in ((1, 'L'), (-1, 'R')):
        r = np.sqrt((xr - sgn * EYE_C[0]) ** 2 + (yr - EYE_C[1] - 0.02) ** 2 + (zr - EYE_C[2] - 0.09) ** 2)
        re = np.sqrt((xr - sgn * EYE_C[0]) ** 2 + (yr - EYE_C[1]) ** 2 + (zr - EYE_C[2]) ** 2)
        w = np.exp(-(r / 0.07) ** 2) * ss(EYE_R + 0.002, EYE_R + 0.012, re) * float(P.get('brow_' + nm, 0))
        dv[:, 1] += -0.006 * w; dv[:, 2] += 0.024 * w
        rc = np.sqrt((xr - sgn * EYE_C[0]) ** 2 + (yr - EYE_C[1] + 0.06) ** 2 + (zr - EYE_C[2] + 0.105) ** 2)
        wc = np.exp(-(rc / 0.06) ** 2) * ss(EYE_R + 0.002, EYE_R + 0.012, re) * float(P.get('echeek_' + nm, 0))
        dv[:, 0] += sgn * 0.004 * wc; dv[:, 1] += -0.008 * wc; dv[:, 2] += 0.016 * wc
    # throat sac: a soft wobble under the chin (spring channel 'jig', signed)
    jt = np.exp(-((xr / 0.15) ** 2 + ((zr - 0.60) / 0.06) ** 2)) * (1 - ss(-0.16, -0.06, yr))
    jig = float(P.get('jig', 0.0))
    dv[:, 1] += -0.006 * jig * jt; dv[:, 2] += -0.004 * jig * jt
    dv *= nb[:, None]
    rest = np.stack([xr, yr, zr], 1)
    vel = np.zeros((len(v), 3))
    # arm verts: texture coords ride with their component; velocity from the neighbouring frames' pose
    for si, S in enumerate(('R', 'L')):
        for ci, nmc in enumerate(names):
            idx = np.where((side == si) & (comp == ci) & (armm > 0.01))[0]
            if not len(idx): continue
            loc = to_local(ARM[S][0], nmc, v[idx].astype(np.float64))
            w = armm[idx][:, None]
            rest[idx] = (1 - w) * rest[idx] + w * (loc + TEX_OFF[nmc])
            if Pn is not None:
                pa = from_local(Pn[0][S][0], nmc, loc); pb = from_local(Pn[1][S][0], nmc, loc)
                vel[idx] = (pb - pa) * 0.5 * FPS * w
    return A, dv.astype(np.float32), rest.astype(np.float32), vel.astype(np.float32)


def side_params(P):
    out = {}
    for S in ('R', 'L'):
        out[S] = arm_pose(P, S)
    return out


def build(P, vol_static, Pprev=None, Pnext=None):
    M = Mouth(P)
    xs, ys, zs = grid()
    vol = vol_static.copy()
    ix0, ix1 = np.searchsorted(xs, BOX['x0']), np.searchsorted(xs, BOX['x1'])
    iz0, iz1 = np.searchsorted(zs, BOX['z0']), np.searchsorted(zs, BOX['z1'])
    iy1 = np.searchsorted(ys, BOX['y1'])
    Y, Z = np.meshgrid(ys[:iy1], zs[iz0:iz1], indexing='ij')
    for i in range(ix0, ix1):
        vol[i, :iy1, iz0:iz1] = M.full(np.full_like(Y, xs[i], dtype=np.float64), Y.astype(np.float64), Z.astype(np.float64))
    ARM = side_params(P)
    for S in ('R', 'L'):
        C = ARM[S][0]
        lo, hi = arm_bbox(C)
        sl, _, (Xg, Yg, Zg) = eval_box(lambda a, b, c: 0, xs, ys, zs, lo, hi)
        da = limb_sdf(C, Xg, Yg, Zg)
        Wp = C['hand'][1][0]
        hlo, hhi = Wp - 0.15, Wp + 0.15
        hm = np.all([(Xg > hlo[0]), (Xg < hhi[0]), (Yg > hlo[1]), (Yg < hhi[1]), (Zg > hlo[2]), (Zg < hhi[2])], 0)
        dh = np.full_like(da, 1.0)
        dh[hm] = hand_sdf(C, Xg[hm], Yg[hm], Zg[hm])
        da = smin(da, dh, 0.014)                                     # wrist blend
        # blend into the body only around the shoulder; elsewhere a near-hard union
        r_sh = np.sqrt((Xg - SHOULDER[S][0]) ** 2 + (Yg - SHOULDER[S][1]) ** 2 + (Zg - SHOULDER[S][2]) ** 2)
        k = 0.002 + 0.045 * np.exp(-(r_sh / 0.075) ** 2)
        vol[sl] = smin(vol[sl].astype(np.float64), da, k).astype(np.float32)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(VOX, VOX, VOX))
    v += np.array([X0, Y0, Z0], np.float32)
    v = taubin(v, f)
    Pn = (side_params(Pprev), side_params(Pnext)) if Pprev is not None else None
    A, dv, rest, vel = rig_attrs(M, v, P, ARM, Pn)
    return v + dv, f.astype(np.int32), A, rest, vel


KEYS = ['smile', 'smirk', 'J', 'U', 'wide', 'pucker', 'fv', 'mbp', 'brow_L', 'brow_R', 'echeek_L', 'echeek_R', 'jig']
ARMK = ['w', 'ha', 'hn', 'cu', 'sp', 'th', 'pole']


def params(anim, fr):
    fr = min(max(fr, 1), len(anim['J']))
    P = {k: anim[k][fr - 1] for k in KEYS if k in anim}
    for S in ('R', 'L'):
        for k in ARMK:
            if k + S in anim: P[k + S] = anim[k + S][fr - 1]
    return P


if __name__ == '__main__':
    argv = sys.argv[1:]
    def opt(k, d=None):
        return argv[argv.index(k) + 1] if k in argv else d
    anim = json.load(open(os.path.join(HERE, opt('--anim', 'anim3.json'))))
    out = os.path.join(HERE, opt('--out', 'meshes3')); os.makedirs(out, exist_ok=True)
    frames = [int(a) for a in opt('--list', '1').split(',')]
    t0 = time.time(); vs = static_volume(); print('static', round(time.time() - t0, 1), flush=True)
    for fr in frames:
        t0 = time.time()
        P = params(anim, fr)
        if '--set' in argv:
            P.update(json.loads(opt('--set')))
        v, f, A, rest, vel = build(P, vs, params(anim, fr - 1), params(anim, fr + 1))
        np.savez(os.path.join(out, f'body_{fr:04d}.npz'), verts=v.astype(np.float32), faces=f, rest=rest, velocity=vel, **A)
        print('MESH', fr, len(v), round(time.time() - t0, 1), flush=True)
