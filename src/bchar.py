# Blender side of the shared froggy rig: materials (locked canon colours), eyes + lids, bandana, jetpack,
# space suit + helmet, and per-frame update from mesh_frames.py npz files.
import bpy, math, os, numpy as np
from mathutils import Vector, Matrix
from blib import *

GROUND = -0.365
EYE_C = (0.165, -0.05, 0.952 - GROUND); EYE_R = 0.098

# LOCKED CANON colours (refs/froggies_v2_locked_colors.jpg + LOCKED_CANON.md)
CANON = {
    'JAMES':   dict(body='#4FA834', spot='#2F7A22', acc='bandana', acc_col='#D3232B'),   # green, red bandana
    'JIMMY':   dict(body='#F9C700', spot='#E09A00', acc='jetpack', acc_col='#F2F3F5'),   # bright yellow, white jetpack
    'BUBBLES': dict(body='#1E5BD8', spot='#123C9C', acc='bandana', acc_col='#F4C20D'),   # true blue, yellow bandana
    'REXY':    dict(body='#F27A12', spot='#C2550A', acc='bandana', acc_col='#1F55C9'),   # orange, blue bandana
}
CREAM = '#F5EACB'; PAD = '#F28A1E'


def skin_material(name, body, spot, suit=False):
    BODY = srgb(body); SPOT = srgb(spot)
    m = principled(name, BODY, rough=0.55, sss=0.25, sss_r=(1.0, 0.55, 0.25), sss_scale=0.03, coat=0.06, coat_r=0.4, spec=0.4)
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    A = lambda n, y: node(nt, 'ShaderNodeAttribute', (-1500, y), attribute_name=n)
    rest = A('rest', 300); cream = A('cream', 0); mouth = A('mouthin', -200); deep = A('mouthdeep', -400)
    tng = A('tongue', -600); lipa = A('lipline', -800); pad = A('pad', -1000)
    # speckles (voronoi on rest coords), big blotchy spots + fine freckles, faded over the cream patch
    vz = node(nt, 'ShaderNodeTexVoronoi', (-1250, 400)); vz.inputs['Scale'].default_value = 16.0; vz.inputs['Randomness'].default_value = 0.9
    nt.links.new(rest.outputs['Vector'], vz.inputs['Vector'])
    rmp = node(nt, 'ShaderNodeValToRGB', (-1050, 400))
    rmp.color_ramp.elements[0].position = 0.0; rmp.color_ramp.elements[0].color = (1, 1, 1, 1)
    rmp.color_ramp.elements[1].position = 0.20; rmp.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(vz.outputs['Distance'], rmp.inputs['Fac'])
    vz2 = node(nt, 'ShaderNodeTexVoronoi', (-1250, 650)); vz2.inputs['Scale'].default_value = 45.0
    nt.links.new(rest.outputs['Vector'], vz2.inputs['Vector'])
    rmp2 = node(nt, 'ShaderNodeValToRGB', (-1050, 650))
    rmp2.color_ramp.elements[0].color = (0.6, 0.6, 0.6, 1); rmp2.color_ramp.elements[1].position = 0.18; rmp2.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(vz2.outputs['Distance'], rmp2.inputs['Fac'])
    spots = node(nt, 'ShaderNodeMath', (-850, 500), operation='MAXIMUM')
    nt.links.new(rmp.outputs['Color'], spots.inputs[0]); nt.links.new(rmp2.outputs['Color'], spots.inputs[1])
    nocr = node(nt, 'ShaderNodeMath', (-850, 300), operation='SUBTRACT'); nocr.inputs[0].default_value = 1.0
    nt.links.new(cream.outputs['Fac'], nocr.inputs[1])
    nocr2 = node(nt, 'ShaderNodeMath', (-700, 300), operation='POWER'); nocr2.inputs[1].default_value = 3.0
    nt.links.new(nocr.outputs[0], nocr2.inputs[0])
    sm = node(nt, 'ShaderNodeMath', (-550, 450), operation='MULTIPLY'); sm.use_clamp = True
    nt.links.new(spots.outputs[0], sm.inputs[0]); nt.links.new(nocr2.outputs[0], sm.inputs[1])
    base = node(nt, 'ShaderNodeMix', (-400, 450), data_type='RGBA'); base.inputs['A'].default_value = BODY; base.inputs['B'].default_value = SPOT
    nt.links.new(sm.outputs[0], base.inputs['Factor'])
    nlg = node(nt, 'ShaderNodeTexNoise', (-1250, 150)); nlg.inputs['Scale'].default_value = 5.0; nlg.inputs['Detail'].default_value = 3.0
    nt.links.new(rest.outputs['Vector'], nlg.inputs['Vector'])
    vr = node(nt, 'ShaderNodeMapRange', (-1000, 150)); vr.inputs['To Min'].default_value = 0.92; vr.inputs['To Max'].default_value = 1.05
    nt.links.new(nlg.outputs['Fac'], vr.inputs['Value'])
    var = node(nt, 'ShaderNodeMix', (-250, 450), data_type='RGBA', blend_type='MULTIPLY'); var.inputs['Factor'].default_value = 1.0
    nt.links.new(base.outputs['Result'], var.inputs['A'])
    comb = node(nt, 'ShaderNodeCombineColor', (-450, 150))
    for i in range(3): nt.links.new(vr.outputs['Result'], comb.inputs[i])
    nt.links.new(comb.outputs['Color'], var.inputs['B'])
    cm = node(nt, 'ShaderNodeMix', (-100, 400), data_type='RGBA'); cm.inputs['B'].default_value = srgb(CREAM)
    nt.links.new(var.outputs['Result'], cm.inputs['A']); nt.links.new(cream.outputs['Fac'], cm.inputs['Factor'])
    pm = node(nt, 'ShaderNodeMix', (50, 400), data_type='RGBA'); pm.inputs['B'].default_value = srgb(PAD)
    nt.links.new(cm.outputs['Result'], pm.inputs['A']); nt.links.new(pad.outputs['Fac'], pm.inputs['Factor'])
    last = pm
    if suit:
        sw = A('suitw', -1200); sg = A('suitg', -1300)
        s1 = node(nt, 'ShaderNodeMix', (150, 300), data_type='RGBA'); s1.inputs['B'].default_value = srgb('#F2F2EE')
        nt.links.new(last.outputs['Result'], s1.inputs['A']); nt.links.new(sw.outputs['Fac'], s1.inputs['Factor'])
        s2 = node(nt, 'ShaderNodeMix', (250, 300), data_type='RGBA'); s2.inputs['B'].default_value = srgb('#8D939A')
        nt.links.new(s1.outputs['Result'], s2.inputs['A']); nt.links.new(sg.outputs['Fac'], s2.inputs['Factor'])
        last = s2
    mc1 = node(nt, 'ShaderNodeMix', (-100, -500), data_type='RGBA')
    mc1.inputs['A'].default_value = srgb('#B5404C'); mc1.inputs['B'].default_value = srgb('#2E080C')
    nt.links.new(deep.outputs['Fac'], mc1.inputs['Factor'])
    mc2 = node(nt, 'ShaderNodeMix', (50, -500), data_type='RGBA')
    nt.links.new(mc1.outputs['Result'], mc2.inputs['A']); mc2.inputs['B'].default_value = srgb('#E57A84')
    nt.links.new(tng.outputs['Fac'], mc2.inputs['Factor'])
    mm = node(nt, 'ShaderNodeMix', (350, 200), data_type='RGBA')
    nt.links.new(last.outputs['Result'], mm.inputs['A']); nt.links.new(mc2.outputs['Result'], mm.inputs['B'])
    nt.links.new(mouth.outputs['Fac'], mm.inputs['Factor'])
    lm = node(nt, 'ShaderNodeMix', (500, 200), data_type='RGBA', blend_type='MULTIPLY'); lm.inputs['B'].default_value = (0.55, 0.42, 0.35, 1)
    nt.links.new(mm.outputs['Result'], lm.inputs['A'])
    lf = node(nt, 'ShaderNodeMath', (350, -150), operation='MULTIPLY'); lf.inputs[1].default_value = 0.6
    nt.links.new(lipa.outputs['Fac'], lf.inputs[0]); nt.links.new(lf.outputs[0], lm.inputs['Factor'])
    nt.links.new(lm.outputs['Result'], b.inputs['Base Color'])
    rr = node(nt, 'ShaderNodeMapRange', (350, -300)); rr.inputs['To Min'].default_value = 0.55; rr.inputs['To Max'].default_value = 0.35
    nt.links.new(mouth.outputs['Fac'], rr.inputs['Value']); nt.links.new(rr.outputs['Result'], b.inputs['Roughness'])
    bn = node(nt, 'ShaderNodeTexNoise', (-300, -700)); bn.inputs['Scale'].default_value = 140.0; bn.inputs['Detail'].default_value = 2.0
    nt.links.new(rest.outputs['Vector'], bn.inputs['Vector'])
    bump = node(nt, 'ShaderNodeBump', (200, -700)); bump.inputs['Distance'].default_value = 0.002; bump.inputs['Strength'].default_value = 0.12
    nt.links.new(bn.outputs['Fac'], bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def eye_material(name='Eye'):
    if name in bpy.data.materials: return bpy.data.materials[name]
    m = principled(name, srgb('#FBF8F0'), rough=0.25, sss=0.1, sss_scale=0.01, coat=1.0, coat_r=0.02)
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    tc = node(nt, 'ShaderNodeTexCoord', (-1400, 0))
    sep = node(nt, 'ShaderNodeSeparateXYZ', (-1200, 0)); nt.links.new(tc.outputs['Object'], sep.inputs[0])
    ln = node(nt, 'ShaderNodeVectorMath', (-1200, -200), operation='LENGTH'); nt.links.new(tc.outputs['Object'], ln.inputs[0])
    dv = node(nt, 'ShaderNodeMath', (-1000, 0), operation='DIVIDE'); nt.links.new(sep.outputs['Y'], dv.inputs[0]); nt.links.new(ln.outputs['Value'], dv.inputs[1])
    ac = node(nt, 'ShaderNodeMath', (-850, 0), operation='ARCCOSINE'); nt.links.new(dv.outputs[0], ac.inputs[0])
    ang = node(nt, 'ShaderNodeMath', (-700, 0), operation='SUBTRACT'); ang.inputs[0].default_value = math.pi; nt.links.new(ac.outputs[0], ang.inputs[1])
    az = node(nt, 'ShaderNodeMath', (-1000, -350), operation='ARCTAN2'); nt.links.new(sep.outputs['Z'], az.inputs[0]); nt.links.new(sep.outputs['X'], az.inputs[1])
    cmb = node(nt, 'ShaderNodeCombineXYZ', (-850, -350)); nt.links.new(az.outputs[0], cmb.inputs['X']); nt.links.new(ang.outputs[0], cmb.inputs['Y'])
    nz = node(nt, 'ShaderNodeTexNoise', (-650, -350)); nz.inputs['Scale'].default_value = 6.0; nz.inputs['Detail'].default_value = 4.0
    mp = node(nt, 'ShaderNodeMapping', (-700, -550)); mp.inputs['Scale'].default_value = (5.0, 0.6, 1.0)
    nt.links.new(cmb.outputs[0], mp.inputs['Vector']); nt.links.new(mp.outputs[0], nz.inputs['Vector'])
    cr = node(nt, 'ShaderNodeValToRGB', (-450, 0))
    E = cr.color_ramp.elements; E[0].position = 0.0; E[0].color = srgb('#050302')
    E[1].position = 0.30; E[1].color = srgb('#0B0603')
    for p, c in ((0.315, '#7A3A08'), (0.40, '#E8901C'), (0.52, '#F2A630'), (0.59, '#C66A12'), (0.625, '#3A1E08'), (0.65, '#F6F1E6'), (1.0, '#FBF8F0')):
        e = E.new(p); e.color = srgb(c)
    nr = node(nt, 'ShaderNodeMath', (-550, 0), operation='DIVIDE'); nr.inputs[1].default_value = 1.05
    nt.links.new(ang.outputs[0], nr.inputs[0])
    pert = node(nt, 'ShaderNodeMath', (-450, -250), operation='MULTIPLY_ADD'); pert.inputs[1].default_value = 0.045; pert.inputs[2].default_value = -0.022
    nt.links.new(nz.outputs['Fac'], pert.inputs[0])
    addp = node(nt, 'ShaderNodeMath', (-350, -100), operation='ADD'); nt.links.new(nr.outputs[0], addp.inputs[0]); nt.links.new(pert.outputs[0], addp.inputs[1])
    nt.links.new(addp.outputs[0], cr.inputs['Fac'])
    nt.links.new(cr.outputs['Color'], b.inputs['Base Color'])
    return m


def cloth_material(name, col):
    m = principled(name, srgb(col), rough=0.78, sss=0.05, spec=0.25)
    m.node_tree.nodes['Principled BSDF'].inputs['Sheen Weight'].default_value = 0.4
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    w = node(nt, 'ShaderNodeTexWave', (-600, -300)); w.inputs['Scale'].default_value = 140.0; w.inputs['Distortion'].default_value = 1.0
    tc = node(nt, 'ShaderNodeTexCoord', (-800, -300)); nt.links.new(tc.outputs['Object'], w.inputs['Vector'])
    bump = node(nt, 'ShaderNodeBump', (-300, -300)); bump.inputs['Strength'].default_value = 0.08; bump.inputs['Distance'].default_value = 0.001
    nt.links.new(w.outputs['Fac'], bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def glass_material(name='HelmetGlass'):
    if name in bpy.data.materials: return bpy.data.materials[name]
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    tr = node(nt, 'ShaderNodeBsdfTransparent', (0, 100))
    gl = node(nt, 'ShaderNodeBsdfGlossy', (0, -100)); gl.inputs['Roughness'].default_value = 0.03
    lw = node(nt, 'ShaderNodeLayerWeight', (-200, 250)); lw.inputs['Blend'].default_value = 0.12
    mx = node(nt, 'ShaderNodeMixShader', (250, 0))
    nt.links.new(lw.outputs['Fresnel'], mx.inputs[0]); nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
    o = node(nt, 'ShaderNodeOutputMaterial', (450, 0)); nt.links.new(mx.outputs[0], o.inputs[0])
    return m


def lid_angles(up_close, low_raise=0.0):
    return -61 + up_close * 73.0, 26 - low_raise * 28.0


class Froggy:
    def __init__(self, name, suit=False, col=None):
        self.name = name; C = CANON[name]; self.suit = suit
        self.skin = skin_material(name + 'Skin', C['body'], C['spot'], suit)
        self.lidm = principled(name + 'Lid', srgb(C['body']), rough=0.55, sss=0.2, sss_scale=0.02, spec=0.4)
        if suit:
            self.lidm = self.skin_lid = self.lidm
        self.body = link(bpy.data.objects.new(name + '_Body', bpy.data.meshes.new(name + '_tmp')), col)
        self.root = link(bpy.data.objects.new(name + '_Root', None), col)
        self.E = {b: link(bpy.data.objects.new(f'{name}_{b}', None), col) for b in ('head', 'chest', 'jet', 'hand_L', 'hand_R', 'hips')}
        for e in self.E.values(): e.parent = self.root
        self.body.parent = self.root
        eyem = eye_material()
        self.eyes = {}; self.lids = {}
        for side, s in (('L', 1), ('R', -1)):
            piv = link(bpy.data.objects.new(f'{name}_EyePiv_{side}', None), col); piv.parent = self.E['head']
            piv.location = (s * EYE_C[0], EYE_C[1], EYE_C[2])
            eye = uv_sphere(f'{name}_Eye_{side}', EYE_R, 48, 24, eyem, col); eye.parent = piv
            eye.rotation_mode = 'QUATERNION'
            self.eyes[side] = eye
            for part, ang, down in (('up', 96, False), ('low', 74, True)):
                lp = link(bpy.data.objects.new(f'{name}_LidP_{part}_{side}', None), col); lp.parent = piv; lp.rotation_mode = 'XYZ'
                lid = cap_shell(f'{name}_Lid_{part}_{side}', EYE_R + 0.0024, ang, 0.0028, down, self.lidm, col); lid.parent = lp
                self.lids[(part, side)] = lp
        self.acc = {}
        if C['acc'] == 'bandana':
            self.bandm = cloth_material(name + 'Bandana', C['acc_col'])
            self.acc['bandana'] = link(bpy.data.objects.new(name + '_Bandana', bpy.data.meshes.new('b')), col)
            self.acc['bandana'].parent = self.root
        if C['acc'] == 'jetpack':
            self.build_jetpack(col)
        if suit:
            self.build_suit(col)

    def build_jetpack(self, col):
        white = principled('JetWhite', srgb('#EEEFF1'), rough=0.3, coat=0.4, coat_r=0.08)
        grey = principled('JetGrey', srgb('#9EA3A8'), rough=0.35, metal=0.3)
        metal = principled('JetMetal', srgb('#B8BCC0'), rough=0.22, metal=1.0)
        red = principled('JetRed', srgb('#E0471E'), rough=0.35, coat=0.3)
        self.strapm = principled('Strap', srgb('#5D636B'), rough=0.55)
        self.jet_light = principled('JetLight', srgb('#30FF60'), emit=srgb('#30FF60'), emit_s=4.0)
        S = -GROUND; J = self.E['jet']; parts = []
        parts.append(box('JetPlate', (0, 0.235, 0.48 + S), (0.44, 0.07, 0.30), white, 0.03, c=col))
        for s in (-1, 1):
            parts.append(ellipsoid(f'JetTank{s}', (s * 0.265, 0.27, 0.50 + S), (0.092, 0.092, 0.225), white, c=col))
            parts.append(cyl(f'JetCollar{s}', (s * 0.265, 0.27, 0.728 + S), 0.05, 0.034, grey, c=col))
            parts.append(ellipsoid(f'JetCap{s}', (s * 0.265, 0.27, 0.748 + S), (0.032, 0.032, 0.018), grey, c=col))
            parts.append(cyl(f'JetNozzle{s}', (s * 0.265, 0.27, 0.268 + S), 0.052, 0.05, metal, r2=0.06, c=col))
            parts.append(cyl(f'JetRing{s}', (s * 0.265, 0.27, 0.238 + S), 0.056, 0.024, red, c=col))
            for zb in (0.43, 0.57):
                parts.append(cyl(f'JetBand{s}{zb}', (s * 0.265, 0.27, zb + S), 0.0905, 0.026, red, c=col))
        parts.append(box('JetButton', (0.0, 0.272, 0.58 + S), (0.07, 0.02, 0.05), red, 0.008, c=col))
        parts.append(ellipsoid('JetLamp', (0.0, 0.27, 0.42 + S), (0.018, 0.012, 0.018), self.jet_light, c=col))
        # flames (hidden unless thrusting): emissive cones under the nozzles
        self.flames = []
        fm = bpy.data.materials.new('Flame'); fm.use_nodes = True; nt = fm.node_tree; nt.nodes.clear()
        em = node(nt, 'ShaderNodeEmission'); em.inputs['Color'].default_value = srgb('#FFB040'); em.inputs['Strength'].default_value = 25
        tr = node(nt, 'ShaderNodeBsdfTransparent'); lw = node(nt, 'ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.5
        mx = node(nt, 'ShaderNodeMixShader'); nt.links.new(lw.outputs['Facing'], mx.inputs[0]); nt.links.new(em.outputs[0], mx.inputs[1]); nt.links.new(tr.outputs[0], mx.inputs[2])
        o = node(nt, 'ShaderNodeOutputMaterial'); nt.links.new(mx.outputs[0], o.inputs[0])
        for s in (-1, 1):
            fl = cyl(f'Flame{s}', (s * 0.265, 0.27, 0.10 + S), 0.05, 0.28, fm, r2=0.012, rot=(math.pi, 0, 0), c=col, smooth=False)
            fl.rotation_euler = (0, 0, 0)
            # cone points down: radius1 at bottom -> flip by scaling z
            fl.scale = (1, 1, -1)
            fl.hide_render = True; self.flames.append(fl); parts.append(fl)
        for p in parts:
            p.parent = J; p.matrix_parent_inverse = Matrix.Identity(4)
        self.jet_parts = parts
        self.acc['straps'] = link(bpy.data.objects.new(self.name + '_Straps', bpy.data.meshes.new('s')), col)
        self.acc['straps'].parent = self.root
        buck = []
        for s in (-1, 1):
            buck.append(box(f'Buckle{s}', (s * 0.118, -0.215, 0.43 + S), (0.05, 0.02, 0.04), red, 0.008, c=col))
        for p in buck:
            p.parent = self.E['chest']
        self.buckles = buck

    def build_suit(self, col):
        S = -GROUND
        g = glass_material()
        h = cap_shell(self.name + '_Helmet', 0.375, 118, 0.006, False, g, col)
        h.parent = self.E['head']; h.location = (0, -0.025, 0.775 + S)
        ring = principled('SuitRing', srgb('#B9BEC4'), rough=0.3, metal=0.8)
        bpy.ops.mesh.primitive_torus_add(major_radius=0.33, minor_radius=0.03, major_segments=64, minor_segments=12, location=(0, -0.025, 0.585 + S))
        r = bpy.context.active_object; r.name = self.name + '_Collar'; r.data.materials.append(ring); smooth_obj(r)
        r.parent = self.E['chest']
        self.helmet = h

    # ---------------------------------------------------------------- per-frame
    def set_frame(self, npz_path, rootM, face, gaze_world, thrust=0.0):
        D = np.load(npz_path)
        old = self.body.data
        me = mesh_np(self.name + '_BodyMesh', D['verts'], D['faces'], self.skin, obj=False)
        add_attr(me, 'rest', D['rest'])
        for k in ('cream', 'mouthin', 'mouthdeep', 'tongue', 'lipline', 'pad', 'suitw', 'suitg'):
            if k in D.files: add_attr(me, k, D[k])
        self.body.data = me
        if old.users == 0: bpy.data.meshes.remove(old)
        self.root.matrix_world = rootM
        for b, e in self.E.items():
            e.matrix_parent_inverse = Matrix.Identity(4)
            e.matrix_basis = np_to_matrix(D['G_' + b])
        for a, ob in self.acc.items():
            if 'acc_' + a + '_v' not in D.files: continue
            o2 = ob.data
            ob.data = mesh_np(f'{self.name}_{a}', D['acc_' + a + '_v'], D['acc_' + a + '_f'],
                              self.bandm if a == 'bandana' else self.strapm, obj=False)
            if o2.users == 0: bpy.data.meshes.remove(o2)
        bpy.context.view_layer.update()
        # eyes: aim at the gaze target (clamped), lids from blink/squint
        up, low = lid_angles(float(face.get('blink', 0.0)) * 1.0 + float(face.get('lidsq', 0.0)) * 0.35,
                             float(face.get('lowlid', 0.0)))
        for side in ('L', 'R'):
            piv = self.eyes[side].parent
            Mw = piv.matrix_world
            d_local = (Mw.inverted().to_3x3() @ (Vector(gaze_world) - Mw.translation)).normalized()
            yaw = math.atan2(d_local.x, -d_local.y); pitch = -math.atan2(d_local.z, math.hypot(d_local.x, d_local.y))
            yaw = max(-0.6, min(0.6, yaw)); pitch = max(-0.45, min(0.45, pitch))
            q = (Matrix.Rotation(yaw, 3, 'Z') @ Matrix.Rotation(pitch, 3, 'X')).to_quaternion()
            self.eyes[side].rotation_quaternion = q
            self.lids[('up', side)].rotation_euler = (math.radians(up), 0, 0)
            self.lids[('low', side)].rotation_euler = (math.radians(low), 0, 0)
        if hasattr(self, 'flames'):
            for fl in self.flames:
                fl.hide_render = thrust <= 0.01
                fl.scale = (1, 1, -max(0.2, thrust))


class Bear:
    """Purple Bear (she) for the phone video-call shots."""
    def __init__(self, col=None):
        self.name = 'PURPLE_BEAR'
        m = principled('BearFur', srgb('#9B6FD6'), rough=0.75, sss=0.15, sss_scale=0.03, spec=0.3)
        nt = m.node_tree; b = nt.nodes['Principled BSDF']; b.inputs['Sheen Weight'].default_value = 0.6
        b.inputs['Sheen Tint'].default_value = srgb('#E8D8FF')
        A = lambda n, y: node(nt, 'ShaderNodeAttribute', (-900, y), attribute_name=n)
        mz, bl, mi = A('muzzle', 0), A('belly', -200), A('mouthin', -400)
        rest = A('rest', 200)
        nz = node(nt, 'ShaderNodeTexNoise', (-900, 400)); nz.inputs['Scale'].default_value = 60; nz.inputs['Detail'].default_value = 3
        nt.links.new(rest.outputs['Vector'], nz.inputs['Vector'])
        bp = node(nt, 'ShaderNodeBump', (-200, -600)); bp.inputs['Strength'].default_value = 0.25; bp.inputs['Distance'].default_value = 0.004
        nt.links.new(nz.outputs['Fac'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
        c1 = node(nt, 'ShaderNodeMix', (-500, 100), data_type='RGBA'); c1.inputs['A'].default_value = srgb('#9B6FD6'); c1.inputs['B'].default_value = srgb('#E9DAF7')
        mx = node(nt, 'ShaderNodeMath', (-700, 0), operation='MAXIMUM'); nt.links.new(mz.outputs['Fac'], mx.inputs[0]); nt.links.new(bl.outputs['Fac'], mx.inputs[1])
        nt.links.new(mx.outputs[0], c1.inputs['Factor'])
        c2 = node(nt, 'ShaderNodeMix', (-300, 100), data_type='RGBA'); c2.inputs['B'].default_value = srgb('#5A1E2E')
        nt.links.new(c1.outputs['Result'], c2.inputs['A']); nt.links.new(mi.outputs['Fac'], c2.inputs['Factor'])
        nt.links.new(c2.outputs['Result'], b.inputs['Base Color'])
        self.skin = m
        self.root = link(bpy.data.objects.new('PB_Root', None), col)
        self.body = link(bpy.data.objects.new('PB_Body', bpy.data.meshes.new('pbtmp')), col); self.body.parent = self.root
        self.E = {'head': link(bpy.data.objects.new('PB_head', None), col)}; self.E['head'].parent = self.root
        eye = principled('BearEye', srgb('#140C1C'), rough=0.08, coat=1.0, coat_r=0.02)
        self.eyes = {}
        for k, s in (('L', 1), ('R', -1)):
            e = ellipsoid('PB_Eye' + k, (s * 0.105, -0.262, 1.13), (0.034, 0.026, 0.042), eye, c=col); e.parent = self.E['head']
            hl = ellipsoid('PB_EyeHL' + k, (s * 0.105 + 0.012, -0.285, 1.145), (0.008, 0.004, 0.008), emission_mat('HL', (1, 1, 1, 1), 3.0), c=col); hl.parent = e
            self.eyes[k] = e
            lash = cyl('PB_Lash' + k, (s * 0.128, -0.262, 1.168), 0.004, 0.035, eye, rot=(0, math.radians(-s * 55), 0), c=col); lash.parent = self.E['head']
        nose = ellipsoid('PB_Nose', (0, -0.345, 1.04), (0.045, 0.032, 0.03), principled('BearNose', srgb('#3A1E4A'), rough=0.2, coat=0.6), c=col)
        nose.parent = self.E['head']
        bow = principled('Bow', srgb('#FF6FAE'), rough=0.5, sheen=None) if False else principled('Bow', srgb('#FF6FAE'), rough=0.5)
        for s in (-1, 1):
            w = ellipsoid('PB_Bow', (-0.21 + s * 0.06, -0.02, 1.43), (0.06, 0.025, 0.04), bow, rot=(0, s * 0.3, 0), c=col); w.parent = self.E['head']
        k = ellipsoid('PB_BowK', (-0.21, -0.03, 1.43), (0.022, 0.022, 0.022), bow, c=col); k.parent = self.E['head']
        self.acc = {}

    def set_frame(self, npz_path, rootM, face, gaze_world, thrust=0.0):
        D = np.load(npz_path)
        old = self.body.data
        me = mesh_np('PB_BodyMesh', D['verts'], D['faces'], self.skin, obj=False)
        for k in ('rest', 'muzzle', 'belly', 'mouthin'):
            add_attr(me, k, D[k])
        self.body.data = me
        if old.users == 0: bpy.data.meshes.remove(old)
        self.root.matrix_world = rootM
        self.E['head'].matrix_basis = np_to_matrix(D['G_head'])
        bl = float(face.get('blink', 0))
        for e in self.eyes.values(): e.scale = (0.034, 0.026, 0.042 * max(0.08, 1 - bl))
