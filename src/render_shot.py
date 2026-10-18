# Render frames of one shot.  blender -b -P src/render_shot.py -- --shot shots/X.json --list 1,2 --mesh M --out O
#   [--pct 100] [--samples N] [--assets DIR] [--save x.blend]
import bpy, sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from mathutils import Vector, Matrix
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
from blib import *
import bchar, bsets, bprops

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
SHOT = json.load(open(opt('--shot')))
N = SHOT['frames']
R = SHOT.get('render', {})
ASSETS = opt('--assets', os.path.join(HERE, '..', 'assets'))
MESH = opt('--mesh'); OUT = opt('--out'); os.makedirs(OUT, exist_ok=True)
frames = [int(a) for a in opt('--list').split(',')]
T_SETUP = time.time()

# ---------------------------------------------------------------- set, cast, props
SET = bsets.build(SHOT['set'], SHOT.get('set_opts', {}), ASSETS)
chars = {}
for name, C in SHOT['chars'].items():
    chars[name] = bchar.Froggy(name, suit=C.get('suit', False))
PROPS = bprops.build(SHOT.get('props', {}), ASSETS, chars)

# ---------------------------------------------------------------- camera
cam_d = bpy.data.cameras.new('Cam'); cam = link(bpy.data.objects.new('Cam', cam_d)); scn.camera = cam
cam_d.sensor_width = 36.0; cam_d.clip_start = 0.05; cam_d.clip_end = 3000
CAM = SHOT['camera']

# ---------------------------------------------------------------- render settings
pct = int(opt('--pct', R.get('pct', 100)))
scn.render.engine = 'CYCLES'
scn.cycles.device = 'CPU'
scn.render.resolution_x, scn.render.resolution_y = 1280, 720
scn.render.resolution_percentage = pct
scn.render.fps = 24
scn.cycles.samples = int(opt('--samples', R.get('samples', 32)))
scn.cycles.use_adaptive_sampling = True
scn.cycles.adaptive_threshold = float(R.get('noise', 0.03))
scn.cycles.use_denoising = True
try:
    scn.cycles.denoiser = 'OPENIMAGEDENOISE'
    scn.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
except TypeError:
    scn.cycles.use_denoising = False   # sandbox arm64 build has no OIDN (farm x64 build does)
scn.cycles.max_bounces = 6; scn.cycles.diffuse_bounces = 2; scn.cycles.glossy_bounces = 2
scn.cycles.transmission_bounces = 4; scn.cycles.transparent_max_bounces = 12; scn.cycles.volume_bounces = 0
scn.cycles.caustics_reflective = False; scn.cycles.caustics_refractive = False
scn.cycles.blur_glossy = 1.0
scn.cycles.sample_clamp_indirect = 6.0
scn.cycles.use_auto_tile = False
scn.cycles.seed = 0; scn.cycles.use_animated_seed = False
scn.render.use_persistent_data = True
scn.view_settings.view_transform = 'AgX'
scn.view_settings.look = R.get('look', 'AgX - Medium High Contrast')
scn.view_settings.exposure = float(R.get('exposure', 0.0))
scn.render.film_transparent = False
scn.render.image_settings.file_format = 'PNG'; scn.render.image_settings.color_depth = '8'
scn.render.use_motion_blur = False
try:
    scn.cycles.texture_limit_render = '2048'
except Exception:
    pass


def M_root(r):
    x, y, z, yaw = r
    return Matrix.Translation((x, y, z)) @ Matrix.Rotation(math.radians(yaw), 4, 'Z')


print('SETUP', SHOT['id'], round(time.time() - T_SETUP, 1), flush=True)
t_all = time.time()
for fr in frames:
    if opt('--wait'):
        okf = os.path.join(MESH, f'{fr:04d}.ok')
        while not os.path.exists(okf): time.sleep(0.3)
    t0 = time.time(); i = fr - 1
    scn.frame_set(fr)
    for name, C in SHOT['chars'].items():
        ch = chars[name]
        vis = C.get('visible', [1] * N)[i]
        objs = [ch.body, *ch.eyes.values(), *ch.acc.values()] + [c for e in ch.E.values() for c in e.children_recursive]
        for ob in set(objs): ob.hide_render = not vis
        if not vis: continue
        ch.set_frame(os.path.join(MESH, f'{name}_{fr:04d}.npz'), M_root(C['root'][i]), C['face'][i], C['gaze'][i],
                     C.get('thrust', [0] * N)[i])
        if hasattr(ch, 'flames'):
            for fl in ch.flames: fl.hide_render = (not vis) or C.get('thrust', [0] * N)[i] <= 0.01
    bprops.update(PROPS, SHOT.get('props', {}), i, chars)
    bsets.update(SET, SHOT.get('set_opts', {}), i, fr)
    p, t = CAM['pos'][i], CAM['tgt'][i]
    cam.matrix_world = look_at_matrix(p, t, CAM.get('roll', [0] * N)[i])
    cam_d.lens = CAM['lens'][i]
    fd = CAM.get('focus', [None] * N)[i]
    if fd:
        cam_d.dof.use_dof = True; cam_d.dof.focus_distance = fd; cam_d.dof.aperture_fstop = CAM.get('fstop', 2.8)
    else:
        cam_d.dof.use_dof = False
    scn.render.filepath = os.path.join(OUT, f"{SHOT['id']}_{fr:04d}.png")
    if opt('--save') and fr == frames[0]:
        bpy.ops.wm.save_as_mainfile(filepath=opt('--save'))
    bpy.ops.render.render(write_still=True)
    print('RENDERED', SHOT['id'], fr, round(time.time() - t0, 1), flush=True)
    if opt('--wait'):
        for name in SHOT['chars']:
            p_ = os.path.join(MESH, f'{name}_{fr:04d}.npz')
            if os.path.exists(p_): os.remove(p_)
        os.remove(okf)
print('ALLDONE', round(time.time() - t_all, 1))
