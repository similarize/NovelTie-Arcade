# Blender helpers shared by the movie scripts (Blender 4.3).
import bpy, bmesh, math, numpy as np
from mathutils import Vector, Matrix


def srgb(h, a=1.0):
    h = h.lstrip('#'); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c) + (a,)


def coll(name=None):
    return bpy.context.scene.collection


def link(ob, c=None):
    (c or bpy.context.scene.collection).objects.link(ob); return ob


def smooth_obj(ob):
    for p in ob.data.polygons: p.use_smooth = True
    return ob


def node(nt, t, loc=(0, 0), **kw):
    n = nt.nodes.new(t); n.location = loc
    for k, v in kw.items():
        if k in n.inputs: n.inputs[k].default_value = v
        else: setattr(n, k, v)
    return n


def principled(name, color, rough=0.5, sss=0.0, sss_r=(1, 0.5, 0.3), sss_scale=0.02, coat=0.0, coat_r=0.1,
               spec=0.5, metal=0.0, emit=None, emit_s=0.0, alpha=1.0, trans=0.0, ior=1.45):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = color if len(color) == 4 else tuple(color) + (1,)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Subsurface Weight'].default_value = sss
    b.inputs['Subsurface Radius'].default_value = sss_r
    b.inputs['Subsurface Scale'].default_value = sss_scale
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Coat Roughness'].default_value = coat_r
    b.inputs['Specular IOR Level'].default_value = spec
    b.inputs['Metallic'].default_value = metal
    b.inputs['Transmission Weight'].default_value = trans
    b.inputs['IOR'].default_value = ior
    b.inputs['Alpha'].default_value = alpha
    if emit is not None:
        b.inputs['Emission Color'].default_value = emit if len(emit) == 4 else tuple(emit) + (1,)
        b.inputs['Emission Strength'].default_value = emit_s
    return m


def emission_mat(name, color, strength=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    e = node(nt, 'ShaderNodeEmission'); e.inputs['Color'].default_value = color; e.inputs['Strength'].default_value = strength
    o = node(nt, 'ShaderNodeOutputMaterial', (300, 0)); nt.links.new(e.outputs[0], o.inputs[0])
    return m


def mesh_np(name, V, F, mat=None, smooth=True, obj=True, c=None):
    me = bpy.data.meshes.new(name)
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int32)
    me.vertices.add(len(V)); me.vertices.foreach_set('co', V.ravel())
    me.loops.add(F.size); me.loops.foreach_set('vertex_index', F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', np.arange(0, F.size, F.shape[1], dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(len(F), F.shape[1], np.int32)) if F.shape[1] != 3 else None
    me.update(calc_edges=True); me.validate(clean_customdata=False)
    if smooth: me.shade_smooth()
    if mat: me.materials.append(mat)
    if not obj: return me
    return link(bpy.data.objects.new(name, me), c)


def add_attr(me, name, arr):
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        at = me.attributes.new(name, 'FLOAT_VECTOR', 'POINT'); at.data.foreach_set('vector', arr.ravel())
    else:
        at = me.attributes.new(name, 'FLOAT', 'POINT'); at.data.foreach_set('value', arr)


def uv_sphere(name, r, seg=48, ring=24, mat=None, c=None):
    m = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=ring, radius=r)
    bm.to_mesh(m); bm.free()
    ob = link(bpy.data.objects.new(name, m), c); smooth_obj(ob)
    if mat: ob.data.materials.append(mat)
    return ob


def cap_shell(name, r, max_angle_deg, thick, down=False, mat=None, c=None):
    m = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=36, radius=r)
    ca = math.cos(math.radians(max_angle_deg))
    kill = [v for v in bm.verts if ((-v.co.z if down else v.co.z) / r) < ca - 1e-6]
    bmesh.ops.delete(bm, geom=kill, context='VERTS')
    bm.to_mesh(m); bm.free()
    ob = link(bpy.data.objects.new(name, m), c); smooth_obj(ob)
    s = ob.modifiers.new('Solidify', 'SOLIDIFY'); s.thickness = thick; s.offset = 1.0; s.use_rim = True
    sd = ob.modifiers.new('Subsurf', 'SUBSURF'); sd.levels = 1; sd.render_levels = 1
    if mat: ob.data.materials.append(mat)
    return ob


def box(name, loc, size, mat=None, bevel=0.0, rot=(0, 0, 0), sub=0, c=None):
    m = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    bm.to_mesh(m); bm.free()
    ob = link(bpy.data.objects.new(name, m), c); ob.location = loc; ob.rotation_euler = rot
    if bevel > 0:
        bv = ob.modifiers.new('Bevel', 'BEVEL'); bv.width = bevel; bv.segments = 3; bv.use_clamp_overlap = True
        bv.harden_normals = True
        smooth_obj(ob)
    if sub:
        ob.modifiers.new('Sub', 'SUBSURF').levels = sub; smooth_obj(ob)
    if mat: ob.data.materials.append(mat)
    return ob


def cyl(name, loc, r, depth, mat=None, rot=(0, 0, 0), seg=32, r2=None, bevel=0.0, c=None, smooth=True):
    m = bpy.data.meshes.new(name); bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r if r2 is None else r2, depth=depth)
    bm.to_mesh(m); bm.free()
    ob = link(bpy.data.objects.new(name, m), c); ob.location = loc; ob.rotation_euler = rot
    if bevel > 0:
        bv = ob.modifiers.new('Bevel', 'BEVEL'); bv.width = bevel; bv.segments = 2
    if smooth:
        smooth_obj(ob)
        # keep caps flat-looking: edge split on sharp angles
        es = ob.modifiers.new('ES', 'EDGE_SPLIT'); es.split_angle = math.radians(50)
    if mat: ob.data.materials.append(mat)
    return ob


def ellipsoid(name, loc, scale, mat=None, rot=(0, 0, 0), seg=32, c=None):
    ob = uv_sphere(name, 1.0, seg, seg // 2, mat, c); ob.location = loc; ob.scale = scale; ob.rotation_euler = rot
    return ob


def np_to_matrix(M):
    return Matrix([list(map(float, r)) for r in np.asarray(M)])


def look_at_matrix(eye, target, roll=0.0):
    eye = Vector(eye); d = (Vector(target) - eye).normalized()
    q = d.to_track_quat('-Z', 'Y')
    M = q.to_matrix().to_4x4()
    if roll: M = M @ Matrix.Rotation(math.radians(roll), 4, 'Z')
    M.translation = eye
    return M
