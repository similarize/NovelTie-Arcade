# Reusable sets: ranch (lawn, garden, house + porch, garage + Cybertruck, monster-truck track, pond, mechs,
# Starship pad, forest edge), glowing forest (7-colour orbs + golden orb), space (Earth, Mars, stars),
# Purple Bear's home. Units: froggy = 1.42 tall. Ranch layout (z up, camera usually looks +y):
#   lawn lineup (0,0) | garden beds x -14..-4, y 1..8 | house centre (0,24) porch front y=17.5 | garage (22,24)
#   pond (14,-7) r6 | track oval (-34,-8) | mechs (-26,38),(-17,41) | Starship pad (46,70) | forest edge y>50
import bpy, bmesh, math, random, os, numpy as np
from mathutils import Vector, Matrix
from blib import *

LAYOUT = dict(house=(0, 24), porch_y=17.5, garage=(22, 24), pond=(14, -7, 6.0), track=(-34, -8, 15, 8),
              mechs=[(-26, 38), (-17, 41)], pad=(46, 70), garden=(-9, 4.5), forest_y=52)
_C = {}


def world(hdri, strength=1.0, rot=0.0, sun=None, bg_strength=None, color=None):
    w = bpy.data.worlds.new('World'); bpy.context.scene.world = w; w.use_nodes = True
    try:
        w.cycles.sampling_method = 'MANUAL'; w.cycles.sample_map_resolution = 512
    except Exception:
        pass
    wn = w.node_tree; bg = wn.nodes['Background']
    if hdri:
        env = node(wn, 'ShaderNodeTexEnvironment', (-500, 0)); env.image = bpy.data.images.load(hdri)
        mp = node(wn, 'ShaderNodeMapping', (-700, 0)); tc = node(wn, 'ShaderNodeTexCoord', (-900, 0))
        wn.links.new(tc.outputs['Generated'], mp.inputs['Vector']); wn.links.new(mp.outputs[0], env.inputs['Vector'])
        mp.inputs['Rotation'].default_value = (0, 0, math.radians(rot))
        wn.links.new(env.outputs['Color'], bg.inputs['Color'])
    elif color:
        bg.inputs['Color'].default_value = color
    bg.inputs['Strength'].default_value = strength
    if sun:
        ld = bpy.data.lights.new('Sun', 'SUN'); ld.energy = sun.get('energy', 3.0); ld.angle = math.radians(sun.get('angle', 2.0))
        ld.color = sun.get('color', (1.0, 0.92, 0.8))
        so = link(bpy.data.objects.new('Sun', ld))
        so.rotation_euler = (math.radians(sun.get('elev_from_zenith', 55)), 0, math.radians(sun.get('azim', 30)))
        return so


def find_hdri(assets, name):
    for res in ('4k', '2k', '1k'):
        p = os.path.join(assets, f'{name}_{res}.hdr')
        if os.path.exists(p): return p
    return None


# ---------------------------------------------------------------- materials
def M(name, *a, **k):
    if name in _C: return _C[name]
    _C[name] = principled(name, *a, **k); return _C[name]


def noise_mat(name, c1, c2, scale=2.0, rough=0.8, bump=0.0, detail=4.0, coord='Object'):
    if name in _C: return _C[name]
    m = principled(name, srgb(c1), rough=rough, spec=0.3)
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = node(nt, 'ShaderNodeTexCoord', (-900, 0))
    n = node(nt, 'ShaderNodeTexNoise', (-700, 0)); n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail
    nt.links.new(tc.outputs[coord], n.inputs['Vector'])
    r = node(nt, 'ShaderNodeValToRGB', (-450, 0)); r.color_ramp.elements[0].color = srgb(c1); r.color_ramp.elements[1].color = srgb(c2)
    r.color_ramp.elements[0].position = 0.3; r.color_ramp.elements[1].position = 0.7
    nt.links.new(n.outputs['Fac'], r.inputs['Fac']); nt.links.new(r.outputs['Color'], b.inputs['Base Color'])
    if bump:
        bp = node(nt, 'ShaderNodeBump', (-200, -300)); bp.inputs['Strength'].default_value = bump
        nt.links.new(n.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    _C[name] = m; return m


def wood_mat(name='Wood', c1='#6B4A2C', c2='#A27B52'):
    if name in _C: return _C[name]
    m = principled(name, srgb(c2), rough=0.75)
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    w = node(nt, 'ShaderNodeTexWave', (-500, 0)); w.inputs['Scale'].default_value = 3; w.inputs['Distortion'].default_value = 6
    tc = node(nt, 'ShaderNodeTexCoord', (-700, 0)); nt.links.new(tc.outputs['Object'], w.inputs['Vector'])
    r = node(nt, 'ShaderNodeValToRGB', (-300, 0)); r.color_ramp.elements[0].color = srgb(c1); r.color_ramp.elements[1].color = srgb(c2)
    nt.links.new(w.outputs['Fac'], r.inputs['Fac']); nt.links.new(r.outputs['Color'], b.inputs['Base Color'])
    _C[name] = m; return m


def grass_ground_mat():
    if 'GroundGrass' in _C: return _C['GroundGrass']
    m = principled('GroundGrass', srgb('#4E7A2A'), rough=0.95, spec=0.1)
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = node(nt, 'ShaderNodeTexCoord', (-1000, 0))
    n = node(nt, 'ShaderNodeTexNoise', (-800, 0)); n.inputs['Scale'].default_value = 0.08; n.inputs['Detail'].default_value = 6
    nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    r = node(nt, 'ShaderNodeValToRGB', (-550, 0)); r.color_ramp.elements[0].color = srgb('#3A5E1C'); r.color_ramp.elements[1].color = srgb('#6E8E30')
    nt.links.new(n.outputs['Fac'], r.inputs['Fac'])
    n2 = node(nt, 'ShaderNodeTexNoise', (-800, -300)); n2.inputs['Scale'].default_value = 9.0; n2.inputs['Detail'].default_value = 8
    nt.links.new(tc.outputs['Object'], n2.inputs['Vector'])
    mul = node(nt, 'ShaderNodeMix', (-300, 0), data_type='RGBA', blend_type='MULTIPLY')
    mr = node(nt, 'ShaderNodeMapRange', (-550, -300)); mr.inputs['To Min'].default_value = 0.75; mr.inputs['To Max'].default_value = 1.15
    nt.links.new(n2.outputs['Fac'], mr.inputs['Value'])
    cc = node(nt, 'ShaderNodeCombineColor', (-400, -300))
    for i in range(3): nt.links.new(mr.outputs['Result'], cc.inputs[i])
    mul.inputs['Factor'].default_value = 1.0
    nt.links.new(r.outputs['Color'], mul.inputs['A']); nt.links.new(cc.outputs['Color'], mul.inputs['B'])
    nt.links.new(mul.outputs['Result'], b.inputs['Base Color'])
    bp = node(nt, 'ShaderNodeBump', (-200, -400)); bp.inputs['Strength'].default_value = 0.4; bp.inputs['Distance'].default_value = 0.05
    nt.links.new(n2.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    _C['GroundGrass'] = m; return m


def grass_blade_mat():
    if 'GrassBlades' in _C: return _C['GrassBlades']
    gh = principled('GrassBlades', srgb('#5C8C2E'), rough=0.5, spec=0.3)
    nt2 = gh.node_tree; hb = nt2.nodes['Principled BSDF']
    gt_ = node(nt2, 'ShaderNodeAttribute', (-900, 0), attribute_name='gt')
    gr_ = node(nt2, 'ShaderNodeAttribute', (-900, -200), attribute_name='grnd')
    mx = node(nt2, 'ShaderNodeMath', (-700, 0), operation='MULTIPLY_ADD'); mx.inputs[1].default_value = 0.75
    nt2.links.new(gt_.outputs['Fac'], mx.inputs[0])
    m2 = node(nt2, 'ShaderNodeMath', (-700, -200), operation='MULTIPLY'); m2.inputs[1].default_value = 0.35
    nt2.links.new(gr_.outputs['Fac'], m2.inputs[0]); nt2.links.new(m2.outputs[0], mx.inputs[2])
    ramp = node(nt2, 'ShaderNodeValToRGB', (-450, 0)); E = ramp.color_ramp.elements
    E[0].color = srgb('#2C4A12'); E[1].color = srgb('#B4C85A'); E.new(0.5).color = srgb('#5E8E2A')
    nt2.links.new(mx.outputs[0], ramp.inputs['Fac']); nt2.links.new(ramp.outputs['Color'], hb.inputs['Base Color'])
    _C['GrassBlades'] = gh; return gh


def make_grass(name, cx, cy, rx, ry, n, seed=11, hmin=0.08, hmax=0.26, avoid=None):
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.uniform(0, 1, n)) ; aa = rng.uniform(0, 2 * np.pi, n)
    keep0 = rng.uniform(0, 1, n) < np.clip((1.0 - rr) / 0.35, 0, 1) ** 0.7
    x = cx + rx * rr * np.cos(aa); y = cy + ry * rr * np.sin(aa)
    x, y = x[keep0], y[keep0]; n = len(x)
    if avoid:
        keep = np.ones(n, bool)
        for (ax, ay, ar) in avoid: keep &= (x - ax) ** 2 + (y - ay) ** 2 > ar ** 2
        x, y = x[keep], y[keep]; n = len(x)
    h = rng.uniform(hmin, hmax, n); w = rng.uniform(0.008, 0.015, n)
    yaw = rng.uniform(0, 2 * np.pi, n); bend = rng.uniform(0.15, 0.7, n); bdir = rng.uniform(0, 2 * np.pi, n)
    L = 4; t = np.linspace(0, 1, L)
    cxs, sxs = np.cos(yaw), np.sin(yaw)
    verts = np.zeros((n, L, 2, 3), np.float32)
    for j, tj in enumerate(t):
        hw = w * (1 - tj) ** 0.8 * 0.5 + 0.0004
        off = bend * h * tj ** 2
        bx, by = x + np.cos(bdir) * off, y + np.sin(bdir) * off
        z = h * tj * (1 - 0.25 * bend * tj)
        verts[:, j, 0] = np.stack([bx - cxs * hw, by - sxs * hw, z], 1)
        verts[:, j, 1] = np.stack([bx + cxs * hw, by + sxs * hw, z], 1)
    V2 = verts.reshape(-1, 3)
    base = (np.arange(n) * L * 2)[:, None]
    q = np.array([[j * 2, j * 2 + 1, j * 2 + 3, j * 2 + 2] for j in range(L - 1)])
    faces = (base[:, None, :] + q[None, :, :]).reshape(-1, 4)
    m = bpy.data.meshes.new(name)
    m.vertices.add(len(V2)); m.vertices.foreach_set('co', V2.ravel())
    m.loops.add(faces.size); m.loops.foreach_set('vertex_index', faces.ravel().astype(np.int32))
    m.polygons.add(len(faces)); m.polygons.foreach_set('loop_start', np.arange(0, faces.size, 4, dtype=np.int32))
    m.update(calc_edges=True); m.shade_smooth()
    tt = np.broadcast_to(t[None, :, None], (n, L, 2)).ravel().astype(np.float32)
    rr = np.broadcast_to(rng.uniform(0, 1, n)[:, None, None], (n, L, 2)).ravel().astype(np.float32)
    m.attributes.new('gt', 'FLOAT', 'POINT').data.foreach_set('value', tt)
    m.attributes.new('grnd', 'FLOAT', 'POINT').data.foreach_set('value', rr)
    m.materials.append(grass_blade_mat())
    return link(bpy.data.objects.new(name, m))


# ---------------------------------------------------------------- reusable models
def proto_collection(name):
    c = bpy.data.collections.new(name)
    return c


def instance(col, loc, rot=0.0, scale=1.0, name='inst'):
    e = bpy.data.objects.new(name, None); e.instance_type = 'COLLECTION'; e.instance_collection = col
    e.location = loc; e.rotation_euler = (0, 0, rot); e.scale = (scale, scale, scale)
    return link(e)


def blob_mesh(name, r, seed, sub=3, amp=0.18, mat=None, c=None):
    me = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=r)
    rnd = random.Random(seed); ph = [rnd.uniform(0, 6) for _ in range(6)]
    for v in bm.verts:
        n = v.co.normalized()
        v.co *= 1 + amp * (0.5 * math.sin(n.x * 4 + ph[0]) * math.cos(n.y * 3 + ph[1]) + 0.3 * math.sin(n.z * 5 + ph[2]) + 0.2 * math.sin(n.x * 9 + n.y * 7 + ph[3]))
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); smooth_obj(ob)
    if mat: me.materials.append(mat)
    (c or bpy.context.scene.collection).objects.link(ob)
    return ob


def leaf_mat(name='Leaves', c1='#2F5E1E', c2='#6E9A34'):
    if name in _C: return _C[name]
    m = noise_mat(name, c1, c2, scale=3.0, rough=0.7, bump=0.6, detail=6)
    m.node_tree.nodes['Principled BSDF'].inputs['Subsurface Weight'].default_value = 0.15
    return m


def tree_protos():
    if 'trees' in _C: return _C['trees']
    protos = []
    bark = noise_mat('Bark', '#4A3424', '#6E5038', scale=6, rough=0.9, bump=0.5)
    for k in range(4):
        c = bpy.data.collections.new(f'TreeProto{k}')
        rnd = random.Random(k * 7 + 1)
        h = rnd.uniform(5.5, 8.5)
        tr = cyl(f'Trunk{k}', (0, 0, h * 0.35), 0.28, h * 0.7, bark, r2=0.18, c=c)
        lm = leaf_mat(f'Leaves{k%2}', ['#2F5E1E', '#3B6B22'][k % 2], ['#6E9A34', '#88AA40'][k % 2])
        for j in range(5):
            r = rnd.uniform(1.6, 2.4)
            b = blob_mesh(f'Crown{k}_{j}', r, k * 10 + j, 3, 0.2, lm, c)
            b.location = (rnd.uniform(-1.2, 1.2), rnd.uniform(-1.2, 1.2), h * 0.7 + rnd.uniform(0, 2.2))
        protos.append(c)
    # pines
    pine_m = leaf_mat('Pine', '#1F4A22', '#3E6E2E')
    for k in range(2):
        c = bpy.data.collections.new(f'PineProto{k}')
        h = 9 + 3 * k
        cyl(f'PTrunk{k}', (0, 0, 1.0), 0.25, 2.0, bark, c=c)
        for j in range(4):
            cn = cyl(f'PCone{k}{j}', (0, 0, 2.0 + j * h * 0.18), 2.4 - j * 0.5, h * 0.35, pine_m, r2=0.15, seg=16, c=c)
        protos.append(c)
    _C['trees'] = protos
    return protos


def scatter_trees(region_fn, n, seed, scale=(0.8, 1.3)):
    protos = tree_protos(); rnd = random.Random(seed); placed = 0; tries = 0
    while placed < n and tries < n * 30:
        tries += 1
        p = region_fn(rnd)
        if p is None: continue
        instance(rnd.choice(protos), (p[0], p[1], 0), rnd.uniform(0, 6.28), rnd.uniform(*scale), 'Tree')
        placed += 1


def flower_protos():
    if 'flowers' in _C: return _C['flowers']
    out = []
    for name, ph, ch, n, pl, h in (('Cosmos', '#F07AAE', '#F2C230', 8, 0.07, 0.6), ('Daisy', '#FFFFFF', '#F2B81E', 12, 0.05, 0.45),
                                   ('Aster', '#9A6BDC', '#F4C842', 10, 0.045, 0.5), ('Poppy', '#E8402A', '#2A1A10', 6, 0.065, 0.5)):
        c = bpy.data.collections.new(name + 'Col')
        stem = M('Stem', srgb('#3F6B22'), rough=0.6)
        pm = M(name + 'Petal', srgb(ph), rough=0.45, sss=0.3, sss_scale=0.01)
        cm = M(name + 'Ctr', srgb(ch), rough=0.7)
        cyl(name + 'Stem', (0, 0, h / 2), 0.006, h, stem, seg=6, c=c, smooth=False)
        ellipsoid(name + 'C', (0, 0, h), (0.02, 0.02, 0.011), cm, seg=12, c=c)
        for i in range(n):
            a = 2 * math.pi * i / n
            ellipsoid(name + f'P{i}', (math.cos(a) * pl * 0.6, math.sin(a) * pl * 0.6, h - 0.004), (pl * 0.6, pl * 0.2, 0.004), pm,
                      rot=(0, -0.18, a), seg=10, c=c)
        out.append(c)
    _C['flowers'] = out
    return out


def flower_bed(cx, cy, rx, ry, count, seed, scale=(0.7, 1.3)):
    pr = flower_protos(); rnd = random.Random(seed)
    for i in range(count):
        instance(rnd.choice(pr), (cx + rnd.uniform(-rx, rx), cy + rnd.uniform(-ry, ry), 0), rnd.uniform(0, 6.28), rnd.uniform(*scale), 'Flower')


def sunflower_proto():
    if 'sunflower' in _C: return _C['sunflower']
    c = bpy.data.collections.new('SunflowerCol')
    stem = M('Stem', srgb('#3F6B22'), rough=0.6); pet = M('SunPetal', srgb('#F7C21B'), rough=0.5, sss=0.2); ctr = M('SunCtr', srgb('#4A2A12'), rough=0.9)
    cyl('SFStem', (0, 0, 0.9), 0.025, 1.8, stem, seg=8, c=c)
    hd = link(bpy.data.objects.new('SFHead', None), c); hd.location = (0, -0.05, 1.82); hd.rotation_euler = (math.radians(70), 0, 0)
    ce = ellipsoid('SFC', (0, 0, 0), (0.16, 0.16, 0.05), ctr, seg=16, c=c); ce.parent = hd
    for i in range(16):
        a = 2 * math.pi * i / 16
        p = ellipsoid(f'SFP{i}', (math.cos(a) * 0.22, math.sin(a) * 0.22, 0), (0.1, 0.04, 0.008), pet, rot=(0, 0, a), seg=10, c=c); p.parent = hd
    for j in range(3):
        lf = ellipsoid(f'SFL{j}', (0.12 * (-1) ** j, 0, 0.5 + j * 0.4), (0.16, 0.06, 0.01), stem, rot=(0, 0.3 * (-1) ** j, 0), seg=10, c=c)
    _C['sunflower'] = c; return c


def fence(x0, y0, x1, y1, step=2.2, name='Fence', mat=None, white=False):
    mat = mat or (M('FenceWhite', srgb('#F2F0EA'), rough=0.6) if white else wood_mat())
    L = math.hypot(x1 - x0, y1 - y0); n = max(1, int(L / step)); ang = math.atan2(y1 - y0, x1 - x0)
    for i in range(n + 1):
        t = i / n
        box(f'{name}Post', (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 0.5), (0.14, 0.14, 1.0), mat, 0.02)
    for zr in (0.38, 0.78):
        box(f'{name}Rail', ((x0 + x1) / 2, (y0 + y1) / 2, zr), (L, 0.07, 0.12), mat, 0.015, rot=(0, 0, ang))


# ---------------------------------------------------------------- ranch buildings & machines
def house():
    hx, hy = LAYOUT['house']
    wall = noise_mat('HouseWall', '#F3E9D6', '#E6D8BE', scale=0.6, rough=0.85)
    trim = M('Trim', srgb('#FFFFFF'), rough=0.5)
    roof = noise_mat('Roof', '#8C3A2A', '#6E2C20', scale=2, rough=0.8, bump=0.3)
    win = principled('Window', srgb('#9FC6E0'), rough=0.05, metal=0.0, emit=srgb('#FFE7B0'), emit_s=0.6)
    door = wood_mat('Door', '#6B3A22', '#8A5232')
    W, D, H = 18, 10, 6.5
    box('HouseBody', (hx, hy, H / 2), (W, D, H), wall, 0.05)
    # gable roof (prism)
    me = bpy.data.meshes.new('RoofM')
    V = [(-W / 2 - 0.6, -D / 2 - 0.8, H), (W / 2 + 0.6, -D / 2 - 0.8, H), (W / 2 + 0.6, D / 2 + 0.8, H), (-W / 2 - 0.6, D / 2 + 0.8, H),
         (-W / 2 - 0.6, 0, H + 3.4), (W / 2 + 0.6, 0, H + 3.4)]
    F = [(0, 1, 5, 4), (2, 3, 4, 5), (0, 4, 3), (1, 2, 5)]
    me.from_pydata(V, [], F); me.update()
    ro = link(bpy.data.objects.new('Roof', me)); ro.location = (hx, hy, 0); me.materials.append(roof)
    sol = ro.modifiers.new('Sol', 'SOLIDIFY'); sol.thickness = 0.25
    box('Chimney', (hx + 5.5, hy + 1.5, H + 3.5), (1.0, 1.0, 3.0), noise_mat('Brick', '#9A4A35', '#7A3A2A', 8, 0.9, 0.4), 0.03)
    # windows (two rows) + door on the front (y = hy - D/2)
    fy = hy - D / 2 - 0.03
    for row, z in enumerate((2.0, 4.8)):
        for i, x in enumerate((-7, -4.2, 4.2, 7) if row == 0 else (-7, -4.2, -1.4, 1.4, 4.2, 7)):
            box('Win', (hx + x, fy, z), (1.3, 0.08, 1.5), win)
            box('WinTrim', (hx + x, fy - 0.03, z), (1.5, 0.05, 0.12), trim)
            box('WinTrimB', (hx + x, fy - 0.03, z - 0.82), (1.6, 0.12, 0.12), trim)
    box('Door', (hx, fy, 1.25), (1.4, 0.1, 2.5), door, 0.02)
    # porch: deck, posts, roof, steps, railing
    py = LAYOUT['porch_y']
    deck = wood_mat('Deck', '#7A5636', '#9C7450')
    box('PorchDeck', (hx, (py + fy) / 2, 0.35), (14, fy - py, 0.7), deck, 0.03)
    for i, x in enumerate(np.linspace(-6.8, 6.8, 6)):
        box('PorchPost', (hx + x, py + 0.2, 2.15), (0.22, 0.22, 3.0), trim, 0.02)
    box('PorchRoof', (hx, (py + fy) / 2 - 0.2, 3.75), (15, fy - py + 1.0, 0.25), roof, 0.03)
    for s in range(3):
        box('Step', (hx, py - 0.35 - s * 0.35, 0.12 + (2 - s) * 0.12 - 0.0), (3.0, 0.4, 0.24 + (2 - s) * 0.24), deck, 0.02)
    for x0, x1 in ((-6.8, -1.6), (1.6, 6.8)):
        box('PorchRail', (hx + (x0 + x1) / 2, py + 0.2, 1.55), (x1 - x0, 0.1, 0.1), trim, 0.01)
        for x in np.arange(x0, x1, 0.35):
            box('Baluster', (hx + x, py + 0.2, 1.1), (0.05, 0.05, 0.85), trim)
    # porch swing + a little bench + flower pots
    box('Bench', (hx - 4.5, fy - 0.6, 1.05), (2.2, 0.6, 0.12), deck, 0.02)
    for s in (-1, 1):
        box('BenchLeg', (hx - 4.5 + s * 0.95, fy - 0.6, 0.85), (0.1, 0.5, 0.35), deck)
    pot = M('Pot', srgb('#B5603A'), rough=0.8)
    for x in (-1.9, 1.9):
        cyl('Pot', (hx + x, py - 1.4, 0.3), 0.35, 0.6, pot, r2=0.45)
        b = blob_mesh('PotPlant', 0.45, int(x * 10), 2, 0.25, leaf_mat())
        b.location = (hx + x, py - 1.4, 0.85)


def garage():
    gx, gy = LAYOUT['garage']
    wall = noise_mat('GarageWall', '#E9E4DA', '#D8D0C2', scale=0.5, rough=0.85)
    roof = M('GarageRoof', srgb('#5A5F66'), rough=0.6, metal=0.3)
    doorm = noise_mat('GarageDoor', '#CFCBC3', '#BDB8AE', scale=1.5, rough=0.6)
    W, D, H = 14, 11, 5.5
    box('Garage', (gx, gy, H / 2), (W, D, H), wall, 0.05)
    box('GarageRoof', (gx, gy, H + 0.2), (W + 0.6, D + 0.6, 0.4), roof, 0.05)
    for x in (-3.4, 3.4):
        box('GDoor', (gx + x, gy - D / 2 - 0.03, 1.9), (5.2, 0.1, 3.8), doorm, 0.02)
        for z in np.arange(0.6, 3.8, 0.6):
            box('GDoorLine', (gx + x, gy - D / 2 - 0.09, z), (5.1, 0.03, 0.04), M('GLine', srgb('#A8A39A'), rough=0.6))
    # driveway
    box('Drive', (gx, gy - D / 2 - 5, 0.01), (12, 10, 0.02), noise_mat('Concrete', '#B8B4AC', '#9E9A92', 3, 0.9, 0.2))


def cybertruck(loc, rot=0.0):
    steel = principled('CTSteel', srgb('#B9BDC2'), rough=0.28, metal=1.0)
    glass = principled('CTGlass', srgb('#1A1E24'), rough=0.05, metal=0.2)
    tyre = M('Tyre', srgb('#1B1B1D'), rough=0.85)
    root = link(bpy.data.objects.new('Cybertruck', None)); root.location = loc; root.rotation_euler = (0, 0, rot)
    me = bpy.data.meshes.new('CT')
    L, Wd = 5.8, 2.2
    pr = [(-L / 2, 0.55), (-L / 2, 1.05), (-L * 0.08, 1.95), (L / 2, 1.15), (L / 2, 0.55)]
    V = [(x, -Wd / 2, z) for x, z in pr] + [(x, Wd / 2, z) for x, z in pr]
    n = len(pr)
    F = [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    me.from_pydata(V, [], F); me.update()
    body = link(bpy.data.objects.new('CTBody', me)); body.parent = root; me.materials.append(steel)
    bv = body.modifiers.new('B', 'BEVEL'); bv.width = 0.03; bv.segments = 2
    g = box('CTGlass', (-L * 0.15, 0, 1.5), (1.9, Wd + 0.02, 0.5), glass, rot=(0, math.radians(-15), 0)); g.parent = root
    for x in (-1.9, 1.9):
        for y in (-1.0, 1.0):
            t = cyl('CTWheel', (x, y, 0.45), 0.45, 0.35, tyre, rot=(math.pi / 2, 0, 0)); t.parent = root
    lb = box('CTLight', (L / 2 - 0.02, 0, 1.1), (0.04, Wd * 0.95, 0.05), emission_mat('CTLED', srgb('#FFFFFF'), 3)); lb.parent = root
    return root


def monster_truck(loc, rot=0.0, name='MonsterTruck'):
    paint = principled(name + 'Paint', srgb('#2B6BE0'), rough=0.25, coat=0.8, coat_r=0.05)
    tyre = noise_mat('MTyre', '#1B1B1D', '#2A2A2C', 20, 0.9, 0.8)
    chrome = principled('Chrome', srgb('#D0D4D8'), rough=0.1, metal=1.0)
    root = link(bpy.data.objects.new(name, None)); root.location = loc; root.rotation_euler = (0, 0, rot)
    b = box(name + 'Body', (0, 0, 2.3), (4.2, 2.2, 1.0), paint, 0.15); b.parent = root
    c = box(name + 'Cab', (-0.3, 0, 3.1), (2.0, 2.0, 0.9), paint, 0.15); c.parent = root
    w = box(name + 'Win', (0.72, 0, 3.15), (0.05, 1.8, 0.6), principled('MTGlass', srgb('#20262E'), rough=0.05)); w.parent = root
    for x in (-1.5, 1.5):
        for y in (-1.35, 1.35):
            t = cyl(name + 'Tyre', (x, y, 1.0), 1.0, 0.8, tyre, rot=(math.pi / 2, 0, 0), seg=40, bevel=0.15); t.parent = root
            h = cyl(name + 'Hub', (x, y * 1.32, 1.0), 0.35, 0.05, chrome, rot=(math.pi / 2, 0, 0)); h.parent = root
    a = box(name + 'Axle', (0, 0, 1.0), (3.4, 2.4, 0.18), chrome); a.parent = root
    return root


def mech(loc, rot, col='#C8CDD3', accent='#E04A2A', name='Mech', s=1.0):
    metal = principled(name + 'Metal', srgb(col), rough=0.35, metal=0.8)
    acc = principled(name + 'Acc', srgb(accent), rough=0.4, coat=0.5)
    dark = M('MechDark', srgb('#3A3F46'), rough=0.5, metal=0.6)
    eye = emission_mat(name + 'Eye', srgb('#6FE8FF'), 6)
    root = link(bpy.data.objects.new(name, None)); root.location = loc; root.rotation_euler = (0, 0, rot); root.scale = (s, s, s)
    def P(o): o.parent = root; return o
    for x in (-1.2, 1.2):
        P(box(name + 'Leg', (x, 0, 2.2), (0.9, 1.1, 4.4), metal, 0.15))
        P(box(name + 'Foot', (x, -0.4, 0.3), (1.3, 2.0, 0.6), dark, 0.12))
        P(cyl(name + 'Knee', (x, -0.1, 3.0), 0.55, 1.0, dark, rot=(0, math.pi / 2, 0)))
    P(box(name + 'Hip', (0, 0, 4.7), (3.4, 1.6, 0.9), dark, 0.15))
    P(box(name + 'Torso', (0, 0, 6.6), (3.8, 2.4, 3.0), metal, 0.3))
    P(box(name + 'Chest', (0, -1.22, 6.8), (2.4, 0.1, 1.4), acc, 0.1))
    P(box(name + 'Head', (0, -0.2, 8.6), (1.6, 1.5, 1.1), metal, 0.25))
    P(box(name + 'Visor', (0, -0.98, 8.65), (1.2, 0.06, 0.32), eye))
    for x in (-2.4, 2.4):
        P(box(name + 'Arm', (x, 0, 6.0), (0.8, 0.9, 3.6), metal, 0.15))
        P(box(name + 'Shoulder', (x, 0, 7.8), (1.3, 1.4, 0.9), acc, 0.2))
        P(box(name + 'Hand', (x, -0.1, 3.9), (0.9, 1.0, 0.8), dark, 0.15))
    return root


def starship(loc, height=48.0, name='Starship'):
    steel = principled('StarshipSteel', srgb('#C9CDD2'), rough=0.22, metal=1.0)
    nt = steel.node_tree; b = nt.nodes['Principled BSDF']
    n = node(nt, 'ShaderNodeTexNoise', (-600, 0)); n.inputs['Scale'].default_value = 0.8; n.inputs['Detail'].default_value = 3
    tc = node(nt, 'ShaderNodeTexCoord', (-800, 0)); nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    mr = node(nt, 'ShaderNodeMapRange', (-400, 0)); mr.inputs['To Min'].default_value = 0.14; mr.inputs['To Max'].default_value = 0.32
    nt.links.new(n.outputs['Fac'], mr.inputs['Value']); nt.links.new(mr.outputs['Result'], b.inputs['Roughness'])
    tile = M('HeatTile', srgb('#1D1F22'), rough=0.6)
    conc = noise_mat('PadConcrete', '#A9A59E', '#8F8B84', 2, 0.9, 0.3)
    root = link(bpy.data.objects.new(name, None)); root.location = loc
    R = 4.5
    def P(o): o.parent = root; return o
    P(cyl('PadDisc', (0, 0, 0.15), 11, 0.3, conc, seg=64))
    P(cyl('Mount', (0, 0, 2.5), R * 1.25, 5.0, M('Mount', srgb('#5A5E64'), rough=0.5, metal=0.5), seg=48))
    body_h = height * 0.72
    P(cyl(name + 'Body', (0, 0, 5 + body_h / 2), R, body_h, steel, seg=64))
    P(cyl(name + 'Tiles', (0, 0.25, 5 + body_h / 2), R * 1.003, body_h, tile, seg=64)).scale = (1, 1, 1)
    # cut the tiles to the windward half: shift so only the back half pokes through
    root.children[-1].location.y = 0.0; root.children[-1].scale = (1.0, 1.0, 1.0)
    root.children[-1].hide_render = True
    nose = P(cyl(name + 'Nose', (0, 0, 5 + body_h + height * 0.13), R, height * 0.26, steel, r2=0.6, seg=64))
    P(ellipsoid(name + 'Tip', (0, 0, 5 + body_h + height * 0.26), (0.9, 0.9, 1.0), steel))
    for s in (-1, 1):
        P(box(name + 'Flap', (s * (R + 0.6), 0, 9), (1.4, 0.25, 6.5), tile, 0.1))
        P(box(name + 'FFlap', (s * (R * 0.55 + 0.4), 0, 5 + body_h + height * 0.13), (1.0, 0.2, 3.6), tile, 0.08, rot=(0, s * 0.3, 0)))
    # launch tower (chopsticks)
    tw = M('Tower', srgb('#3C4046'), rough=0.6, metal=0.5)
    for x in (-1.5, 1.5):
        for y in (-1.5, 1.5):
            P(box('Tower', (R + 6 + x, 4 + y, height * 0.6), (0.4, 0.4, height * 1.2), tw))
    for z in np.arange(4, height * 1.2, 4):
        P(box('TowerX', (R + 6, 4, z), (3.4, 3.4, 0.25), tw))
    return root


def pond(cx, cy, r):
    water = principled('Water', srgb('#3C7FA0'), rough=0.04, spec=0.6, trans=0.0)
    nt = water.node_tree; b = nt.nodes['Principled BSDF']
    wv = node(nt, 'ShaderNodeTexNoise', (-600, -300)); wv.inputs['Scale'].default_value = 3; wv.inputs['Detail'].default_value = 3
    bp = node(nt, 'ShaderNodeBump', (-300, -300)); bp.inputs['Strength'].default_value = 0.08
    nt.links.new(wv.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    me = bpy.data.meshes.new('Pond'); bm = bmesh.new(); bmesh.ops.create_circle(bm, cap_ends=True, radius=r, segments=64)
    for v in bm.verts:
        a = math.atan2(v.co.y, v.co.x); v.co *= 1 + 0.12 * math.sin(3 * a) + 0.06 * math.cos(5 * a)
    bm.to_mesh(me); bm.free()
    p = link(bpy.data.objects.new('PondWater', me)); p.location = (cx, cy, 0.03); me.materials.append(water)
    rock = noise_mat('Rock', '#7D776E', '#9A948A', 4, 0.85, 0.5)
    rnd = random.Random(3)
    for i in range(30):
        a = i / 30 * 6.283; rr = r * (1 + 0.12 * math.sin(3 * a) + 0.06 * math.cos(5 * a)) + 0.2
        b = blob_mesh('PondRock', rnd.uniform(0.25, 0.5), i, 2, 0.3, rock); b.location = (cx + rr * math.cos(a), cy + rr * math.sin(a), 0.05)
        b.scale = (1.3, 1.0, 0.6)
    pad = M('LilyPad', srgb('#4F8A2E'), rough=0.5)
    for i in range(9):
        a = rnd.uniform(0, 6.28); d = rnd.uniform(0.5, r * 0.8)
        cyl('Lily', (cx + d * math.cos(a), cy + d * math.sin(a), 0.05), rnd.uniform(0.25, 0.45), 0.02, pad)
    # fishies (orange) just under / jumping: animated in update()
    fish_m = principled('Fish', srgb('#FF8A1E'), rough=0.3, coat=0.5)
    fish = []
    for i in range(3):
        f = ellipsoid(f'Fish{i}', (cx, cy, -1), (0.32, 0.12, 0.14), fish_m)
        t = cyl(f'FishTail{i}', (0.35, 0, 0), 0.0, 0.2, fish_m, r2=0.14, rot=(0, math.pi / 2, 0)); t.parent = f
        fish.append(f)
    return fish


def track(cx, cy, rx, ry):
    dirt = noise_mat('Dirt', '#7A5A3A', '#9A7652', 1.5, 0.95, 0.6, 6)
    me = bpy.data.meshes.new('Track'); bm = bmesh.new()
    n = 96; outer = []; inner = []
    for i in range(n):
        a = i / n * 6.2832
        outer.append(bm.verts.new((cx + (rx + 2.2) * math.cos(a), cy + (ry + 2.2) * math.sin(a), 0.02)))
        inner.append(bm.verts.new((cx + (rx - 2.2) * math.cos(a), cy + (ry - 2.2) * math.sin(a), 0.02)))
    for i in range(n):
        j = (i + 1) % n; bm.faces.new((outer[i], outer[j], inner[j], inner[i]))
    bm.to_mesh(me); bm.free()
    ob = link(bpy.data.objects.new('Track', me)); me.materials.append(dirt)
    # jump ramps + tyre walls
    for a in (0.0, math.pi):
        x, y = cx + rx * math.cos(a), cy + ry * math.sin(a)
        r = box('Ramp', (x, y, 0.6), (3.0, 4.5, 1.2), dirt, 0.5, rot=(0, 0, a + math.pi / 2)); r.scale = (1, 1, 1)
    tyre = M('Tyre', srgb('#1B1B1D'), rough=0.85)
    for i in range(0, n, 6):
        a = i / n * 6.2832
        cyl('TyreWall', (cx + (rx + 3.0) * math.cos(a), cy + (ry + 3.0) * math.sin(a), 0.25), 0.5, 0.5, tyre, seg=16)


def hills():
    me = bpy.data.meshes.new('Hills'); bm = bmesh.new()
    n, rings = 128, 6
    vs = []
    for k in range(rings):
        rr = 160 + k * 60; row = []
        for i in range(n):
            a = i / n * 6.2832
            h = (k > 0) * (8 + 10 * math.sin(a * 3 + k) + 6 * math.sin(a * 7 + 1.3 * k)) * min(1, k / 2)
            row.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), max(0, h))))
        vs.append(row)
    for k in range(rings - 1):
        for i in range(n):
            j = (i + 1) % n; bm.faces.new((vs[k][i], vs[k][j], vs[k + 1][j], vs[k + 1][i]))
    bm.to_mesh(me); bm.free()
    ob = link(bpy.data.objects.new('Hills', me)); smooth_obj(ob)
    me.materials.append(noise_mat('HillGrass', '#4E7A30', '#6F9440', 0.02, 0.95))
    return ob


# ---------------------------------------------------------------- sets
def build_ranch(o, assets):
    sun = world(find_hdri(assets, o.get('hdri', 'qwantani_morning_puresky')), o.get('env', 0.9), o.get('hdri_rot', 0),
                sun=dict(energy=o.get('sun', 3.2), elev_from_zenith=o.get('sun_elev', 62), azim=o.get('sun_azim', -35), color=(1.0, 0.88, 0.72)))
    me = bpy.data.meshes.new('Ground'); bm = bmesh.new(); bmesh.ops.create_circle(bm, cap_ends=True, radius=190, segments=128)
    bm.to_mesh(me); bm.free()
    g = link(bpy.data.objects.new('Ground', me)); me.materials.append(grass_ground_mat())
    hills()
    house(); garage()
    cybertruck((LAYOUT['garage'][0] + 3.5, LAYOUT['garage'][1] - 10, 0), math.radians(90))
    fish = pond(*LAYOUT['pond'])
    cx, cy, rx, ry = LAYOUT['track']; track(cx, cy, rx, ry)
    mt = monster_truck((cx + rx, cy, 0), 0)
    for (x, y), c, a in zip(LAYOUT['mechs'], ('#C8CDD3', '#D9C36A'), ('#E04A2A', '#2A6AE0')):
        mech((x, y, 0), math.radians(-15), c, a, 'Mech%d' % int(x))
    starship((LAYOUT['pad'][0], LAYOUT['pad'][1], 0))
    # garden: raised beds, veggies, sunflowers, white picket fence
    gx, gy = LAYOUT['garden']
    bedwood = wood_mat('BedWood', '#7A5230', '#9A6C42')
    soil = noise_mat('Soil', '#3E2A1C', '#5A3E2A', 6, 0.95, 0.6)
    veg = [M('Pumpkin', srgb('#F08A24'), rough=0.5), M('Cabbage', srgb('#7FB24A'), rough=0.6, sss=0.1), M('Tomato', srgb('#E0352A'), rough=0.35, coat=0.4)]
    rnd = random.Random(5)
    for k, (bx, by) in enumerate([(-4.5, 2.5), (-4.5, 6.0), (2.5, 6.0), (6.5, 3.5)]):
        box('Bed', (gx + bx, gy + by, 0.3), (3.0, 1.6, 0.6), bedwood, 0.03)
        box('BedSoil', (gx + bx, gy + by, 0.58), (2.8, 1.4, 0.06), soil)
        for j in range(7):
            m = veg[(k + j) % 3]
            v = ellipsoid('Veg', (gx + bx + rnd.uniform(-1.2, 1.2), gy + by + rnd.uniform(-0.55, 0.55), 0.72),
                          (0.18, 0.18, 0.14) if m.name != 'Tomato' else (0.09, 0.09, 0.09), m, seg=16)
            if m.name == 'Cabbage': v.scale = (0.2, 0.2, 0.15)
    sf = sunflower_proto()
    for i in range(12):
        instance(sf, (gx - 6.5 + i * 1.1, gy + 9.0 + rnd.uniform(-0.2, 0.2), 0), rnd.uniform(-0.3, 0.3), rnd.uniform(0.9, 1.2), 'Sunflower')
    fence(gx - 7.5, gy + 10.2, gx + 8.5, gy + 10.2, 1.0, 'Picket', white=True)
    flower_bed(gx - 1, gy + 8.4, 6.5, 0.6, 260, 3)
    flower_bed(-5.5, -4.0, 2.5, 1.0, 120, 7)
    flower_bed(6.0, 2.5, 1.5, 2.5, 110, 9)
    # lawn rail fence around the front + grass blades where the froggies act
    fence(-40, -22, 40, -22, 2.4, 'FrontFence')
    for nm, cx_, cy_, rx_, ry_, n in o.get('grass', [['GrassLawn', 0, 0, 11, 9, 360000], ['GrassGarden', -9, 4.5, 9, 6, 240000]]):
        make_grass(nm, cx_, cy_, rx_, ry_, n, seed=hash(nm) % 1000,
                   avoid=[(gx + bx, gy + by, 1.7) for bx, by in [(-4.5, 2.5), (-4.5, 6.0), (2.5, 6.0), (6.5, 3.5)]])
    # trees: forest edge to the north + scattered trees
    fy = LAYOUT['forest_y']
    scatter_trees(lambda r: (r.uniform(-90, 30), r.uniform(fy, fy + 40)), o.get('ntrees', 160), 1)
    scatter_trees(lambda r: (lambda x, y: (x, y) if (abs(x) > 30 or y < -30) and math.hypot(x, y) < 140 else None)(r.uniform(-140, 140), r.uniform(-120, 50)), 120, 2)
    # pines just behind the garden
    return dict(sun=sun, fish=fish, truck=mt)


def build_forest(o, assets):
    sun = world(find_hdri(assets, o.get('hdri', 'qwantani_morning_puresky')), o.get('env', 0.75), o.get('hdri_rot', 0),
                sun=dict(energy=o.get('sun', 1.6), elev_from_zenith=50, azim=-30, color=(1.0, 0.85, 0.65)))
    moss = noise_mat('ForestFloor', '#2E4A1C', '#4F6A2A', 0.6, 0.95, 0.5)
    me = bpy.data.meshes.new('Ground'); bm = bmesh.new(); bmesh.ops.create_circle(bm, cap_ends=True, radius=120, segments=64)
    bm.to_mesh(me); bm.free()
    link(bpy.data.objects.new('Ground', me)).data.materials.append(moss)
    # big trunks + canopy
    bark = noise_mat('ForestBark', '#3A2A1E', '#5C4430', 5, 0.9, 0.7)
    canopy = leaf_mat('Canopy', '#1F4418', '#3E6A26')
    rnd = random.Random(21)
    path_clear = lambda x, y: abs(x - 0.6 * math.sin(y * 0.08) * 6) < 3.0 and -20 < y < 60
    for i in range(140):
        x, y = rnd.uniform(-45, 45), rnd.uniform(-25, 80)
        if path_clear(x, y): continue
        h = rnd.uniform(14, 22); r = rnd.uniform(0.5, 1.1)
        cyl('FTrunk', (x, y, h / 2), r, h, bark, r2=r * 0.6, seg=16)
        for j in range(3):
            b = blob_mesh('FCanopy', rnd.uniform(4, 6.5), i * 3 + j, 2, 0.25, canopy)
            b.location = (x + rnd.uniform(-3, 3), y + rnd.uniform(-3, 3), h + rnd.uniform(-1, 3))
    # ferns / bushes / mushrooms + grass
    fern = leaf_mat('Fern', '#2F5E1E', '#5E8E34')
    for i in range(160):
        x, y = rnd.uniform(-30, 30), rnd.uniform(-15, 60)
        if path_clear(x, y) and rnd.random() < 0.8: continue
        b = blob_mesh('Bush', rnd.uniform(0.5, 1.2), 500 + i, 2, 0.3, fern); b.location = (x, y, 0.2); b.scale = (1.2, 1.2, 0.7)
    shroom = M('Shroom', srgb('#E8E0D0'), rough=0.6, emit=srgb('#9AF0FF'), emit_s=0.6)
    for i in range(60):
        x, y = rnd.uniform(-12, 12), rnd.uniform(-10, 45)
        if path_clear(x, y): continue
        ellipsoid('Shroom', (x, y, 0.15), (0.12, 0.12, 0.06), shroom, seg=12)
    L = bpy.data.lights.new('ForestFill', 'AREA'); L.energy = 3000; L.size = 30; L.color = (0.75, 0.9, 1.0)
    lo = link(bpy.data.objects.new('ForestFill', L)); lo.location = (0, 5, 18)
    make_grass('ForestGrass', 0, 6, 9, 14, o.get('ngrass', 180000), seed=4, hmin=0.06, hmax=0.2)
    # orbs: 7 colours bobbing between the trees (+ the golden one, handled as a prop)
    cols = ['#FF3B30', '#FF9500', '#FFE620', '#34C759', '#2F7BFF', '#5E3BFF', '#C04BFF']
    orbs = []
    for i in range(int(o.get('norbs', 42))):
        c = cols[i % 7]
        x, y = rnd.uniform(-12, 12), rnd.uniform(-6, 40)
        z = rnd.uniform(1.2, 4.5)
        ob = uv_sphere(f'Orb{i}', rnd.uniform(0.10, 0.16), 24, 12, emission_mat(f'OrbM{i%7}', srgb(c), 3.5))
        halo = uv_sphere(f'OrbH{i}', ob.dimensions.x * 1.25, 24, 12, holo_like(f'OrbHM{i%7}', c, 2.0)); halo.parent = ob
        ob.location = (x, y, z)
        if i % 3 == 0:
            ld = bpy.data.lights.new(f'OrbL{i}', 'POINT'); ld.energy = 25; ld.color = srgb(c)[:3]; ld.shadow_soft_size = 0.15
            lo = link(bpy.data.objects.new(f'OrbL{i}', ld)); lo.parent = ob
        ob['base'] = (x, y, z); ob['ph'] = rnd.uniform(0, 6.28)
        orbs.append(ob)
    return dict(sun=sun, orbs=orbs)


def build_space(o, assets):
    w = world(None, 1.0, color=(0, 0, 0, 1))
    wn = bpy.context.scene.world.node_tree; bg = wn.nodes['Background']
    # star field: voronoi points as emission in the background
    tc = node(wn, 'ShaderNodeTexCoord'); vz = node(wn, 'ShaderNodeTexVoronoi'); vz.inputs['Scale'].default_value = 400
    wn.links.new(tc.outputs['Generated'], vz.inputs['Vector'])
    rm = node(wn, 'ShaderNodeMapRange'); rm.inputs['From Min'].default_value = 0.04; rm.inputs['From Max'].default_value = 0.0
    rm.inputs['To Min'].default_value = 0; rm.inputs['To Max'].default_value = 3.0
    wn.links.new(vz.outputs['Distance'], rm.inputs['Value'])
    gate = node(wn, 'ShaderNodeMath', operation='GREATER_THAN'); gate.inputs[1].default_value = 0.93
    wn.links.new(vz.outputs['Color'], gate.inputs[0])
    mul = node(wn, 'ShaderNodeMath', operation='MULTIPLY'); wn.links.new(rm.outputs[0], mul.inputs[0]); wn.links.new(gate.outputs[0], mul.inputs[1])
    wn.links.new(mul.outputs[0], bg.inputs['Strength']); bg.inputs['Color'].default_value = (1, 1, 1, 1)
    ld = bpy.data.lights.new('Sun', 'SUN'); ld.energy = o.get('sun', 4.0); ld.angle = math.radians(0.6); ld.color = (1, 0.97, 0.92)
    so = link(bpy.data.objects.new('Sun', ld)); so.rotation_euler = (math.radians(70), 0, math.radians(o.get('sun_azim', 60)))
    # Earth: procedural continents + clouds + atmosphere rim
    er = o.get('earth_r', 60.0)
    em = bpy.data.materials.new('Earth'); em.use_nodes = True; nt = em.node_tree; b = nt.nodes['Principled BSDF']
    tcn = node(nt, 'ShaderNodeTexCoord'); n = node(nt, 'ShaderNodeTexNoise'); n.inputs['Scale'].default_value = 2.2 / er; n.inputs['Detail'].default_value = 8
    nt.links.new(tcn.outputs['Object'], n.inputs['Vector'])
    cr = node(nt, 'ShaderNodeValToRGB'); E = cr.color_ramp.elements
    E[0].position = 0.52; E[0].color = srgb('#123E8C'); E[1].position = 0.55; E[1].color = srgb('#3E7A34')
    E.new(0.62).color = srgb('#8A7A4A'); E.new(0.7).color = srgb('#5E8A3E')
    nt.links.new(n.outputs['Fac'], cr.inputs['Fac'])
    cl = node(nt, 'ShaderNodeTexNoise'); cl.inputs['Scale'].default_value = 5.0 / er; cl.inputs['Detail'].default_value = 6; cl.inputs['Distortion'].default_value = 1.5
    nt.links.new(tcn.outputs['Object'], cl.inputs['Vector'])
    cr2 = node(nt, 'ShaderNodeValToRGB'); cr2.color_ramp.elements[0].position = 0.55; cr2.color_ramp.elements[1].position = 0.72
    nt.links.new(cl.outputs['Fac'], cr2.inputs['Fac'])
    mx = node(nt, 'ShaderNodeMix', data_type='RGBA'); mx.inputs['B'].default_value = (0.95, 0.95, 0.95, 1)
    nt.links.new(cr.outputs[0], mx.inputs['A']); nt.links.new(cr2.outputs[0], mx.inputs['Factor'])
    nt.links.new(mx.outputs['Result'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = 0.6
    lw = node(nt, 'ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.25
    b.inputs['Emission Color'].default_value = srgb('#5FA8FF'); nt.links.new(lw.outputs['Facing'], b.inputs['Emission Strength'])
    earth = uv_sphere('Earth', er, 128, 64, em); earth.location = o.get('earth_pos', (0, 120, -70))
    atm = uv_sphere('Atmo', er * 1.025, 96, 48, holo_like('AtmoM', '#6FB6FF', 1.5)); atm.parent = earth
    # Mars (small red planet)
    mm = noise_mat('Mars', '#B8472A', '#E07A4A', 3, 0.9, 0.3)
    mars = uv_sphere('Mars', o.get('mars_r', 1.2), 64, 32, mm); mars.location = o.get('mars_pos', (-40, 260, 30))
    return dict(earth=earth, mars=mars)


def holo_like(name, col, strength):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    em = node(nt, 'ShaderNodeEmission'); em.inputs['Color'].default_value = srgb(col); em.inputs['Strength'].default_value = strength
    tr = node(nt, 'ShaderNodeBsdfTransparent'); lw = node(nt, 'ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.6
    pw = node(nt, 'ShaderNodeMath', operation='POWER'); pw.inputs[1].default_value = 3.0
    nt.links.new(lw.outputs['Facing'], pw.inputs[0])
    mx = node(nt, 'ShaderNodeMixShader'); nt.links.new(pw.outputs[0], mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    o = node(nt, 'ShaderNodeOutputMaterial'); nt.links.new(mx.outputs[0], o.inputs[0])
    return m


def build_pbhome(o, assets):
    world(None, 0.3, color=srgb('#FFE6C8'))
    wall = noise_mat('PBWall', '#F6D9E6', '#EEC8DA', 1.0, 0.9)
    floor = noise_mat('PBFloor', '#9A6A48', '#B88A62', 3, 0.6, 0.1)
    box('PBFloorB', (0, 0, -0.05), (12, 12, 0.1), floor)
    box('PBBack', (0, 3.5, 2.5), (12, 0.2, 5), wall)
    box('PBLeft', (-4.5, 0, 2.5), (0.2, 12, 5), wall)
    win = principled('PBWin', srgb('#CFE8FF'), emit=srgb('#DDF0FF'), emit_s=3.0)
    box('PBWindow', (1.8, 3.38, 2.4), (1.8, 0.05, 1.6), win)
    for x in (0.8, 2.8):
        box('PBCurtain', (x, 3.3, 2.5), (0.5, 0.08, 2.2), M('Curtain', srgb('#B48AD8'), rough=0.9, sss=0.1))
    sofa = noise_mat('Sofa', '#F2A6C2', '#E890B2', 6, 0.9, 0.2)
    box('Sofa', (-1.0, 2.6, 0.45), (3.0, 1.0, 0.9), sofa, 0.15)
    box('SofaBack', (-1.0, 3.05, 1.1), (3.0, 0.3, 0.9), sofa, 0.15)
    for x in (-2.4, 0.4):
        box('SofaArm', (x, 2.6, 0.75), (0.3, 1.0, 0.7), sofa, 0.12)
    lamp = principled('Lamp', srgb('#FFF2D0'), emit=srgb('#FFE2A0'), emit_s=6)
    cyl('LampPole', (-3.4, 2.8, 0.9), 0.04, 1.8, M('Brass', srgb('#C8A050'), rough=0.3, metal=1.0))
    cyl('LampShade', (-3.4, 2.8, 1.95), 0.35, 0.4, lamp, r2=0.22)
    L = bpy.data.lights.new('PBKey', 'AREA'); L.energy = 400; L.size = 2.5; L.color = (1.0, 0.92, 0.82)
    lo = link(bpy.data.objects.new('PBKey', L)); lo.location = (1.5, -3, 3.5); lo.rotation_euler = (math.radians(55), 0, math.radians(20))
    L2 = bpy.data.lights.new('PBLamp', 'POINT'); L2.energy = 120; L2.color = (1.0, 0.8, 0.55)
    link(bpy.data.objects.new('PBLamp', L2)).location = (-3.4, 2.6, 1.8)
    # framed pictures
    for i, x in enumerate((-2.2, 0.2)):
        box('Frame', (x, 3.38, 2.6), (0.7, 0.05, 0.55), wood_mat())
        box('Pic', (x, 3.35, 2.6), (0.6, 0.03, 0.45), principled(f'Pic{i}', srgb(['#7FC8F0', '#F0C27F'][i]), rough=0.6))
    return {}


SETS = dict(ranch=build_ranch, forest=build_forest, space=build_space, pbhome=build_pbhome)


def build(name, o, assets):
    random.seed(7)
    return SETS[name](o, assets)


def update(S, o, i, fr):
    t = (o.get('t0', 0) + i) / 24.0
    for ob in S.get('orbs', []):
        bx, by, bz = ob['base']; ph = ob['ph']
        ob.location = (bx + 0.15 * math.sin(t * 0.7 + ph), by + 0.15 * math.cos(t * 0.6 + ph), bz + 0.25 * math.sin(t * 1.3 + ph))
    for k, f in enumerate(S.get('fish', [])):
        # a fishy jumps every ~3 s, staggered
        period = 3.1 + 0.4 * k; ph = (t + k * 1.1) % period
        cx, cy, r = LAYOUT['pond']
        if ph < 0.8:
            u = ph / 0.8
            x = cx - 2 + k * 1.6 + 1.6 * u; z = 1.4 * math.sin(math.pi * u) - 0.1
            f.location = (x, cy + (k - 1) * 1.8, z); f.rotation_euler = (0, -math.cos(math.pi * u) * 0.9, 0)
        else:
            f.location = (cx, cy, -1.0)
    tr = S.get('truck')
    if tr is not None:
        cx, cy, rx, ry = LAYOUT['track']; a = t * 0.35 + 0.3
        tr.location = (cx + rx * math.cos(a), cy + ry * math.sin(a), 0)
        tr.rotation_euler = (0, 0, math.atan2(ry * math.cos(a), -rx * math.sin(a)))
