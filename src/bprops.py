# Props: golden orb, phone (+ SPS hologram), simple per-frame transforms (absolute or attached to a froggy bone).
import bpy, math, numpy as np
from mathutils import Vector, Matrix
from blib import *


def orb_material(name, col, strength=6.0):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    em = node(nt, 'ShaderNodeEmission'); em.inputs['Color'].default_value = srgb(col); em.inputs['Strength'].default_value = strength
    gl = node(nt, 'ShaderNodeBsdfGlossy'); gl.inputs['Color'].default_value = srgb(col); gl.inputs['Roughness'].default_value = 0.1
    lw = node(nt, 'ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.35
    mx = node(nt, 'ShaderNodeMixShader')
    nt.links.new(lw.outputs['Facing'], mx.inputs[0]); nt.links.new(em.outputs[0], mx.inputs[1]); nt.links.new(gl.outputs[0], mx.inputs[2])
    o = node(nt, 'ShaderNodeOutputMaterial'); nt.links.new(mx.outputs[0], o.inputs[0])
    return m


def make_orb(name, col, r, strength=6.0, light=True, light_w=30.0):
    ob = uv_sphere(name, r, 32, 16, orb_material(name + 'M', col, strength))
    import bsets
    h = uv_sphere(name + 'Halo', r * 2.0, 24, 12, bsets.glow_mat(name + 'HM', col, 2.0)); h.parent = ob
    if light:
        ld = bpy.data.lights.new(name + 'L', 'POINT'); ld.energy = light_w; ld.color = srgb(col)[:3]; ld.shadow_soft_size = r
        lo = link(bpy.data.objects.new(name + 'L', ld)); lo.parent = ob
        ob['light'] = lo.name
    return ob


def make_phone(name='Phone'):
    body = box(name, (0, 0, 0), (0.075, 0.012, 0.14), principled(name + 'Case', srgb('#2B2F36'), rough=0.3, metal=0.4), 0.008)
    scr_m = bpy.data.materials.new(name + 'Screen'); scr_m.use_nodes = True; nt = scr_m.node_tree; nt.nodes.clear()
    tc = node(nt, 'ShaderNodeTexCoord'); gr = node(nt, 'ShaderNodeTexGradient')
    mp = node(nt, 'ShaderNodeMapping'); mp.inputs['Rotation'].default_value = (0, math.pi / 2, 0)
    nt.links.new(tc.outputs['Generated'], mp.inputs['Vector']); nt.links.new(mp.outputs[0], gr.inputs['Vector'])
    cr = node(nt, 'ShaderNodeValToRGB'); cr.color_ramp.elements[0].color = srgb('#7A4CC2'); cr.color_ramp.elements[1].color = srgb('#2AB3E8')
    nt.links.new(gr.outputs['Fac'], cr.inputs['Fac'])
    em = node(nt, 'ShaderNodeEmission'); em.inputs['Strength'].default_value = 2.5; nt.links.new(cr.outputs[0], em.inputs[0])
    o = node(nt, 'ShaderNodeOutputMaterial'); nt.links.new(em.outputs[0], o.inputs[0])
    scr = box(name + 'Scr', (0, -0.0065, 0), (0.066, 0.001, 0.128), scr_m)
    scr.parent = body
    return body


def holo_material(name, col, strength=3.0, alpha=0.6):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    em = node(nt, 'ShaderNodeEmission'); em.inputs['Color'].default_value = srgb(col); em.inputs['Strength'].default_value = strength
    tr = node(nt, 'ShaderNodeBsdfTransparent')
    mx = node(nt, 'ShaderNodeMixShader'); mx.inputs[0].default_value = alpha
    nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    o = node(nt, 'ShaderNodeOutputMaterial'); nt.links.new(mx.outputs[0], o.inputs[0])
    return m


def make_sps(name='SPS'):
    """Solar Positioning System hologram: sun, orbit rings, Earth, Mars, Jupiter + a yellow blip (Jimmy)."""
    root = link(bpy.data.objects.new(name, None))
    parts = {}
    sun = uv_sphere(name + 'Sun', 0.035, 24, 12, holo_material(name + 'SunM', '#FFC34A', 6, 0.9)); sun.parent = root
    ring_m = holo_material(name + 'RingM', '#5FD6FF', 2.5, 0.55)
    for k, (r, col, pr) in enumerate([(0.09, '#4AA8FF', 0.013), (0.14, '#FF5A3A', 0.010), (0.22, '#F2C27A', 0.022)]):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=0.0016, major_segments=96, minor_segments=6)
        t = bpy.context.active_object; t.name = f'{name}Ring{k}'; t.data.materials.append(ring_m); t.parent = root
        ang = [0.6, 2.4, 4.2][k]
        p = uv_sphere(f'{name}P{k}', pr, 20, 10, holo_material(f'{name}PM{k}', col, 4, 0.9)); p.parent = root
        p.location = (r * math.cos(ang), r * math.sin(ang), 0)
        parts[f'p{k}'] = p
    blip = uv_sphere(name + 'Blip', 0.010, 16, 8, holo_material(name + 'BlipM', '#FFE81A', 12, 1.0)); blip.parent = root
    parts['blip'] = blip
    # projector cone (light beam from the phone)
    cone = cyl(name + 'Beam', (0, 0, -0.10), 0.02, 0.2, holo_material(name + 'BeamM', '#5FD6FF', 1.2, 0.15), r2=0.24, smooth=False)
    cone.parent = root
    parts['beam'] = cone
    root['parts'] = list(p.name for p in parts.values())
    return root, parts


def build(spec, assets, chars):
    P = {}
    for name, S in spec.items():
        kind = S['kind']
        if kind == 'orb':
            P[name] = make_orb(name, S.get('color', '#FFC21A'), S.get('r', 0.08), S.get('strength', 8.0), S.get('light', True), S.get('light_w', 20))
        elif kind == 'phone':
            P[name] = make_phone(name)
        elif kind == 'sps':
            P[name], parts = make_sps(name)
            P[name + '_parts'] = parts
    return P


def update(P, spec, i, chars):
    for name, S in spec.items():
        ob = P.get(name)
        if ob is None: continue
        vis = S.get('visible', None)
        v = 1 if vis is None else vis[i]
        for o in [ob] + list(ob.children_recursive):
            o.hide_render = not v
        if not v: continue
        att = S.get('attach')
        loc = S.get('pos', [[0, 0, 0]])[min(i, len(S.get('pos', [[0, 0, 0]])) - 1)]
        rot = S.get('rot', [[0, 0, 0]])[min(i, len(S.get('rot', [[0, 0, 0]])) - 1)]
        sc = S.get('scale', [1.0])[min(i, len(S.get('scale', [1.0])) - 1)]
        if isinstance(sc, (list, tuple)): sc = sc[0]
        L = Matrix.Translation(loc) @ Matrix.Rotation(math.radians(rot[2]), 4, 'Z') @ Matrix.Rotation(math.radians(rot[1]), 4, 'Y') @ \
            Matrix.Rotation(math.radians(rot[0]), 4, 'X') @ Matrix.Diagonal((sc, sc, sc, 1))
        if att:
            cname, bone = att.split('.')
            ob.matrix_world = chars[cname].E[bone].matrix_world @ L
        else:
            ob.matrix_world = L
        if S['kind'] == 'sps' and 'blip' in S:
            P[name + '_parts']['blip'].location = S['blip'][i]
