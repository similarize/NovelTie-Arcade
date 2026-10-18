# Per-frame froggy meshes for a shot (system python: numpy/scipy/skimage).
#   python3 src/mesh_frames.py --shot shots/S01_03.json --list 1,2,3 --out work/S01_03/mesh
# Writes <out>/<CHAR>_<frame:04d>.npz: verts (frog-local, soles at z=0), faces, rest + colour attrs,
# bone matrices (head/chest/jet/hand_L/hand_R, 4x4, same space), skinned accessory shells.
import sys, os, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mfrog as MF, macc, mbear

SHIFT = np.eye(4); SHIFT[2, 3] = -MF.GROUND
SHIFT_I = np.linalg.inv(SHIFT)
ACC = {'JAMES': ['bandana'], 'BUBBLES': ['bandana'], 'REXY': ['bandana'], 'JIMMY': ['straps']}
_rest = {}
_acc = None


def rest_mesh(face, vox):
    face = MF.quant(face)
    k = (MF.mouth_key(face), round(vox, 4), tuple(round(float(face.get(n, 0)), 2) for n in ('brow_L', 'brow_R', 'echeek_L', 'echeek_R')))
    if k not in _rest:
        if len(_rest) > 40: _rest.pop(next(iter(_rest)))
        _rest[k] = MF.build_rest(face, vox)
    return _rest[k]


def frame_mesh(char, pose, face, vox):
    global _acc
    v, f, A, rest = rest_mesh(face, vox)
    pose = dict(pose)
    feet = pose.pop('feet', None)
    if feet is None: feet = {n: MF.J['an_' + n].tolist() for n in 'LR'}
    if feet != 'free':
        pose = MF.leg_ik(pose, {n: np.asarray(t, float) for n, t in feet.items()}, pose.pop('foot_rot', None))
    G = MF.fk(pose)
    vs = MF.skin(v, A['W'], G)
    vs[:, 2] -= MF.GROUND
    out = dict(verts=vs, faces=f, rest=rest, **{k: A[k] for k in A if k != 'W'})
    for b in ('head', 'chest', 'jet', 'hand_L', 'hand_R', 'hips'):
        out['G_' + b] = (SHIFT @ G[b] @ SHIFT_I).astype(np.float32)
    if _acc is None: _acc = macc.build()
    for a in ACC.get(char, []):
        av, af, aw = _acc[a]
        avs = MF.skin(av, aw, G); avs[:, 2] -= MF.GROUND
        out['acc_' + a + '_v'] = avs; out['acc_' + a + '_f'] = af
    return out


def main():
    argv = sys.argv[1:]
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    shot = json.load(open(opt('--shot')))
    out = opt('--out'); os.makedirs(out, exist_ok=True)
    frames = [int(a) for a in opt('--list').split(',')]
    vox_default = float(shot.get('render', {}).get('vox', 0.004))
    thr = int(opt('--throttle', 0))
    for fr in frames:
        while thr and len([f for f in os.listdir(out) if f.endswith('.ok')]) >= thr:
            time.sleep(0.5)
        t0 = time.time()
        for name, C in shot['chars'].items():
            i = fr - 1
            if not C.get('visible', [1] * shot['frames'])[i]: continue
            vox = float(os.environ.get('VOX_OVERRIDE') or C.get('vox', vox_default))
            if name == 'PURPLE_BEAR':
                F = dict(C['face'][i]); F['head'] = C['pose'][i].get('head', (0, 0, 0)); F['J'] = float(F.get('J', 0)) * 0.45
                v, f, A, Gh = mbear.build(F, vox)
                o = dict(verts=v, faces=f, G_head=Gh, **A)
            else:
                o = frame_mesh(name, C['pose'][i], C['face'][i], vox)
            np.savez(os.path.join(out, f'{name}_{fr:04d}.tmp.npz'), **o)
            os.replace(os.path.join(out, f'{name}_{fr:04d}.tmp.npz'), os.path.join(out, f'{name}_{fr:04d}.npz'))
        open(os.path.join(out, f'{fr:04d}.ok'), 'w').close()
        print('MESH', fr, round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
