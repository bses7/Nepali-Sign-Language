import bpy
import numpy as np
import mathutils
import math
import sys
import os

# ============================================
# PARSE ARGUMENTS PASSED FROM MAIN.PY
# ============================================
# sys.argv contains: [blender, --python, script.py, --, avatar_path]
argv = sys.argv
if "--" in argv:
    AVATAR_PATH = argv[argv.index("--") + 1]
else:
    # Fallback if run manually in Blender
    AVATAR_PATH = "C:/projects/FYP/data/Avatars/avatar.glb"

# ============================================
# CONFIGURATION
# ============================================
NPZ_PATH = os.getenv("NPZ_PATH", "C:/projects/FYP/experiments/generated_output.npz")
OUTPUT_GLB = os.getenv("OUTPUT_GLB", "C:/projects/FYP/experiments/nsl_animation.glb")
FRAME_START = 1
HAND_DATA_SCALE = 5.0 

BEND_DIRECTION = 1.0  
SPLAY_AMOUNT = 0.4   
CURL_STRENGTH = 1.2   

ANIMATE_LEFT_SIDE = False 
IDLE_STRENGTH = 0.03       
IDLE_SPEED = 0.1  

FINGER_MAPPING = {
    'Thumb':  [1, 2, 3, 4],
    'Index':  [5, 6, 7, 8],
    'Middle': [9, 10, 11, 12],
    'Ring':   [13, 14, 15, 16],
    'Pinky':  [17, 18, 19, 20],
}

# ============================================
# MATH UTILS
# ============================================

def np_to_vec(coords):
    """Convert NPZ [x, y, z] to Blender Vector"""
    return mathutils.Vector((coords[0], coords[2], -coords[1]))

def get_curl_factor(pts, finger_indices):
    """Calculates how closed a finger is (0.0 to 1.0)"""
    base = np_to_vec(pts[finger_indices[0]])
    tip = np_to_vec(pts[finger_indices[-1]])
    current_dist = (tip - base).length
    
    straight_length = 0
    for i in range(len(finger_indices)-1):
        p1 = np_to_vec(pts[finger_indices[i]])
        p2 = np_to_vec(pts[finger_indices[i+1]])
        straight_length += (p2 - p1).length
        
    if straight_length < 0.001: return 0.0
    ratio = current_dist / straight_length
    return max(0.0, min(1.0, 1.0 - (ratio - 0.3) / 0.7))

def solve_hand_rotation(p_wrist, p_index_base, p_pinky_base, bone, is_left):
    v_forward = (p_index_base - p_wrist).normalized()
    v_side = (p_pinky_base - p_wrist).normalized()

    if is_left:
        palm_normal = v_side.cross(v_forward).normalized()
    else:
        palm_normal = v_forward.cross(v_side).normalized()

    hand_side = v_forward.cross(palm_normal).normalized()
    palm_normal = hand_side.cross(v_forward).normalized()

    target_matrix = mathutils.Matrix((hand_side, v_forward, palm_normal)).transposed()
    parent_mat = bone.parent.matrix.to_3x3() if bone.parent else mathutils.Matrix.Identity(3)
    target_local = parent_mat.inverted() @ target_matrix

    bone_rest = bone.bone.matrix_local.to_3x3()
    rest_local = (bone.parent.bone.matrix_local.to_3x3().inverted() @ bone_rest) if bone.parent else bone_rest
    base_quat = (rest_local.inverted() @ target_local).to_quaternion()

    YAW_CORRECTION_DEG = -25.0  # tweak: try 15–30
    yaw_angle = math.radians(YAW_CORRECTION_DEG) * (1.0 if is_left else -1.0)
    yaw_correction = mathutils.Quaternion((0, 0, 1), yaw_angle)

    TILT_DEG = -20.0  # reduce from -30, since yaw is doing the heavy lifting
    tilt_angle = math.radians(TILT_DEG) * (1.0 if is_left else -1.0)
    tilt_correction = mathutils.Quaternion((1, 0, 0), tilt_angle)

    return base_quat @ yaw_correction @ tilt_correction

def solve_finger_segment(bone, vec_start, vec_end, curl_factor, is_thumb=False):
    target_dir = (vec_end - vec_start).normalized()
    parent_mat = bone.parent.matrix.to_3x3() if bone.parent else mathutils.Matrix.Identity(3)
    local_target = parent_mat.inverted() @ target_dir

    bone_rest = bone.bone.matrix_local.to_3x3()
    rest_local_mat = (bone.parent.bone.matrix_local.to_3x3().inverted() @ bone_rest) if bone.parent else bone_rest
    local_rest_dir = (rest_local_mat @ mathutils.Vector((0, 1, 0))).normalized()

    raw_quat = local_rest_dir.rotation_difference(local_target)
    euler = raw_quat.to_euler('XYZ')

    if not is_thumb:
        min_bend = -0.15  
        if (euler.x * BEND_DIRECTION) < min_bend:
            euler.x = min_bend * BEND_DIRECTION

        procedural_curl = curl_factor * (math.pi / 2) * BEND_DIRECTION
        euler.x = (euler.x * 0.85) + (procedural_curl * 0.15)  # was 0.6 / 0.4

        euler.y *= 0.01
        euler.z *= SPLAY_AMOUNT
    else:
        if (euler.x * BEND_DIRECTION) < -0.3:
            euler.x = -0.1
        euler.x *= 0.7
        euler.y *= 0.7
        euler.z *= 0.8

    return euler.to_quaternion()

# ============================================
# SMOOTHING
# ============================================
def smooth_quaternion_track(keyframes, window=5):
    """
    Smooths a list of quaternions using a sliding window average.
    Handles quaternion double-cover (flipping) before averaging.
    """
    n = len(keyframes)
    smoothed = []
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        neighbors = keyframes[start:end]
        
        ref = neighbors[0]
        flipped = []
        for q in neighbors:
            if ref.dot(q) < 0:
                flipped.append(q.copy())
                flipped[-1].negate()
            else:
                flipped.append(q)
        
        # Average component-wise then normalize
        avg = mathutils.Quaternion((0, 0, 0, 0))
        for q in flipped:
            avg.w += q.w
            avg.x += q.x
            avg.y += q.y
            avg.z += q.z
        count = len(flipped)
        avg.w /= count
        avg.x /= count
        avg.y /= count
        avg.z /= count
        avg.normalize()
        smoothed.append(avg)
    return smoothed


def smooth_all_bone_tracks(armature, window=5):
    """
    After all keyframes are baked, smooth every bone's quaternion track.
    Also sets interpolation to BEZIER for natural easing.
    """
    if not armature.animation_data or not armature.animation_data.action:
        return
    
    action = armature.animation_data.action
    
    # Group fcurves by bone and component (w, x, y, z)
    bone_curves = {}
    for fc in action.fcurves:
        # fcurve path looks like: pose.bones["BoneName"].rotation_quaternion
        if 'rotation_quaternion' not in fc.data_path:
            continue
        bone_name = fc.data_path.split('"')[1]
        if bone_name not in bone_curves:
            bone_curves[bone_name] = {}
        bone_curves[bone_name][fc.array_index] = fc  # 0=w, 1=x, 2=y, 3=z

    for bone_name, curves in bone_curves.items():
        if len(curves) != 4:
            continue
        
        # Collect all keyframe times (should be same across w/x/y/z)
        times = sorted([kp.co[0] for kp in curves[0].keyframe_points])
        if len(times) < 3:
            continue
        
        # Read quaternons per frame
        raw_quats = []
        for t in times:
            q = mathutils.Quaternion((
                curves[0].evaluate(t),  
                curves[1].evaluate(t),  
                curves[2].evaluate(t),  
                curves[3].evaluate(t), 
            ))
            raw_quats.append(q)
        
        smoothed = smooth_quaternion_track(raw_quats, window=window)
        
        for comp_idx, fc in curves.items():
            for i, kp in enumerate(fc.keyframe_points):
                if comp_idx == 0: val = smoothed[i].w
                elif comp_idx == 1: val = smoothed[i].x
                elif comp_idx == 2: val = smoothed[i].y
                else: val = smoothed[i].z
                kp.co[1] = val
                kp.interpolation = 'BEZIER'  
            fc.update()

    print(f"✅ Smoothed {len(bone_curves)} bone tracks (window={window})")

# ============================================
# MAIN
# ============================================

def run_animation():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(f"🔄 Importing Avatar from: {AVATAR_PATH}")
    bpy.ops.import_scene.gltf(filepath=AVATAR_PATH)

    armature = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    if not armature:
        print("❌ Error: No armature found in the GLB file!")
        return

    data = np.load(NPZ_PATH)
    pose, lh, rh = data['pose'], data['lh'] / HAND_DATA_SCALE, data['rh'] / HAND_DATA_SCALE

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    if armature.animation_data: armature.animation_data_clear()

    for f in range(len(pose)):
        scene_frame = FRAME_START + f

        idle_w = math.sin(f * IDLE_SPEED) * IDLE_STRENGTH
        idle_h = math.cos(f * IDLE_SPEED * 0.7) * IDLE_STRENGTH
        
        arm_mapping = [('LeftArm', 11, 13), ('LeftForeArm', 13, 15), ('RightArm', 12, 14), ('RightForeArm', 14, 16)]
        for b_name, s, e in arm_mapping:
            pb = armature.pose.bones.get(b_name)
            if pb:
                if not ANIMATE_LEFT_SIDE and b_name.startswith('Left'):
                    v_start, v_end = mathutils.Vector((0,0,0)), mathutils.Vector((0.15+idle_w, idle_h, -1.0+idle_w))
                    pb.rotation_quaternion = solve_finger_segment(pb, v_start, v_end, 0, False)
                else:
                    pb.rotation_quaternion = solve_finger_segment(pb, np_to_vec(pose[f][s]), np_to_vec(pose[f][e]), 0, True)
                pb.keyframe_insert("rotation_quaternion", frame=scene_frame)
                

        for prefix, h_pts, is_left in [('Left', lh[f], True), ('Right', rh[f], False)]:
            if is_left and not ANIMATE_LEFT_SIDE: continue 
            if np.all(h_pts == 0): continue
            
            # Wrist
            wrist_pb = armature.pose.bones.get(f'{prefix}Hand')
            if wrist_pb:
                wrist_pb.rotation_quaternion = solve_hand_rotation(
                    np_to_vec(h_pts[0]),   
                    np_to_vec(h_pts[9]),   
                    np_to_vec(h_pts[17]),  
                    wrist_pb, is_left
                )
                wrist_pb.keyframe_insert("rotation_quaternion", frame=scene_frame)
            
            bpy.context.view_layer.update()

            # Fingers
            for f_name, indices in FINGER_MAPPING.items():
                curl = get_curl_factor(h_pts, indices)
                for seg_i in range(1, 4):
                    bone_name = f"{prefix}Hand{f_name}{seg_i}"
                    pb = armature.pose.bones.get(bone_name)
                    if pb:
                        start_pt = np_to_vec(h_pts[indices[seg_i-1]])
                        end_pt = np_to_vec(h_pts[indices[seg_i]])
                        pb.rotation_quaternion = solve_finger_segment(pb, start_pt, end_pt, curl, (f_name == 'Thumb'))
                        pb.keyframe_insert("rotation_quaternion", frame=scene_frame)

    bpy.ops.object.mode_set(mode='OBJECT')
    smooth_all_bone_tracks(armature, window=5)
    
    bpy.ops.export_scene.gltf(filepath=OUTPUT_GLB, export_format='GLB', export_apply=True, export_animations=True)
    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    run_animation()