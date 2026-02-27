import bpy
import json
import mathutils
import math
import os
import sys

# ============================================
# HEADLESS ARGUMENT PARSING
# ============================================
def get_args():
    try:
        idx = sys.argv.index("--")
        return sys.argv[idx+1:]
    except (ValueError, IndexError):
        return []

# ============================================
# YOUR ORIGINAL CONFIGURATION & TUNING
# ============================================
ARMATURE_NAME = "Armature"        
START_FRAME = 1                   
FRAME_SKIP = 1                     
BEND_DIRECTION = 1.0 

BASE_SMOOTHING = 0.6 
THUMB_SMOOTHING = 0.7 
FINGER_TIP_SMOOTHING = 0.8  

FIST_THRESHOLD = 0.30  
MAX_OPEN_THRESHOLD = 0.65  
HINGE_STIFFNESS = 0.5  
PREVENT_HYPEREXTENSION = True
FIST_CURL_BLEND = 0.7  

FINGER_MAPPING = {
    'Thumb1': (0, 1),   'Thumb2': (1, 2),   'Thumb3': (2, 3),   'Thumb4': (3, 4),
    'Index1': (5, 6),   'Index2': (6, 7),   'Index3': (7, 8),   'Index4': (8, 8),
    'Middle1': (9, 10), 'Middle2': (10, 11), 'Middle3': (11, 12), 'Middle4': (12, 12),
    'Ring1': (13, 14),  'Ring2': (14, 15),  'Ring3': (15, 16),  'Ring4': (16, 16),
    'Pinky1': (17, 18), 'Pinky2': (18, 19), 'Pinky3': (19, 20), 'Pinky4': (20, 20),
}
FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# ============================================
# YOUR ORIGINAL SMOOTHER CLASS
# ============================================
class SmartSmoother:
    def __init__(self):
        self.history = {} 
        self.rot_history = {}
    
    def update(self, unique_id, current_vector, alpha=0.5):
        if unique_id not in self.history:
            self.history[unique_id] = current_vector
            return current_vector
        prev_vector = self.history[unique_id]
        z_alpha = 1.0 - (1.0 - alpha) * 0.1  
        smoothed_vector = mathutils.Vector((
            prev_vector[0] + (current_vector[0] - prev_vector[0]) * (1.0 - alpha),
            prev_vector[1] + (current_vector[1] - prev_vector[1]) * (1.0 - z_alpha),
            prev_vector[2] + (current_vector[2] - prev_vector[2]) * (1.0 - alpha)
        ))
        self.history[unique_id] = smoothed_vector
        return smoothed_vector

    def rotation_update(self, unique_id, current_quat, alpha=0.7):
        if unique_id not in self.rot_history:
            self.rot_history[unique_id] = current_quat
            return current_quat
        smoothed_quat = self.rot_history[unique_id].slerp(current_quat, 1.0 - alpha)
        self.rot_history[unique_id] = smoothed_quat
        return smoothed_quat

# ============================================
# YOUR ORIGINAL ROTATION SOLVERS
# ============================================
def mediapipe_to_vector(landmark, scale=2.0):
    return mathutils.Vector(((landmark['x'] - 0.5) * scale, landmark['z'] * scale, -(landmark['y'] - 0.5) * scale))

def get_finger_curl_factor(vec_base, vec_tip, wrist_pos):
    tip_dist = (vec_tip - wrist_pos).length
    base_dist = (vec_base - wrist_pos).length
    if base_dist > 0.01:
        extension_ratio = tip_dist / base_dist
        if extension_ratio > 2.3: return 0.0
        elif extension_ratio < 0.95: return 1.0
        else:
            normalized = (extension_ratio - 0.95) / (2.3 - 0.95)
            return 1.0 - (normalized ** 0.7)
    return 0.0

def get_hand_closed_factor(p_wrist, p_middle_tip):
    dist = (p_middle_tip - p_wrist).length
    if dist < FIST_THRESHOLD: return 1.0
    if dist > MAX_OPEN_THRESHOLD: return 0.0
    return 1.0 - ((dist - FIST_THRESHOLD) / (MAX_OPEN_THRESHOLD - FIST_THRESHOLD))

def solve_ball_joint(bone, vec_start, vec_end):
    target_dir = (vec_end - vec_start).normalized()
    parent_mat = bone.parent.matrix.to_3x3() if bone.parent else mathutils.Matrix.Identity(3)
    local_target = parent_mat.inverted() @ target_dir
    bone_rest = bone.bone.matrix_local.to_3x3()
    if bone.parent:
        parent_rest = bone.parent.bone.matrix_local.to_3x3()
        rest_diff = parent_rest.inverted() @ bone_rest
    else: rest_diff = bone_rest
    local_rest = (rest_diff @ mathutils.Vector((0, 1, 0))).normalized()
    return local_rest.rotation_difference(local_target)

def solve_tracked_finger_segment(bone, vec_start, vec_end, finger_curl_factor, depth, is_thumb=False, is_pinky=False):
    raw_quat = solve_ball_joint(bone, vec_start, vec_end)
    euler = raw_quat.to_euler('XYZ')
    if not is_thumb:
        euler.y *= (1.0 - HINGE_STIFFNESS)
        euler.z *= (1.0 - HINGE_STIFFNESS)
        limit = -0.4 if is_pinky else -0.2
        if PREVENT_HYPEREXTENSION and euler.x < limit: euler.x = limit
        if euler.x > 2.6: euler.x = 2.6
    else:
        if euler.x > 2.4: euler.x = 2.4
        if euler.x < -0.3: euler.x = -0.3
    
    tracked_quat = euler.to_quaternion()
    curl_amount = ({1: 1.0, 2: 1.6, 3: 1.9, 4: 1.8} if not is_thumb else {1: 0.8, 2: 1.4, 3: 1.6, 4: 1.5}).get(depth, 1.6)
    fist_quat = mathutils.Euler((curl_amount * BEND_DIRECTION, 0, 0), 'XYZ').to_quaternion()
    return tracked_quat.slerp(fist_quat, (finger_curl_factor ** 1.2) * FIST_CURL_BLEND)

def solve_hand_rotation(p_wrist, p_index, p_pinky, bone, is_left):
    v_index = (p_index - p_wrist).normalized()
    v_pinky = (p_pinky - p_wrist).normalized()
    palm_normal = v_pinky.cross(v_index).normalized() if is_left else v_index.cross(v_pinky).normalized()
    hand_dir = (v_index + v_pinky).normalized()
    hand_side = hand_dir.cross(palm_normal).normalized()
    palm_normal = hand_side.cross(hand_dir).normalized()
    target_matrix = mathutils.Matrix((hand_side, hand_dir, palm_normal)).transposed()
    parent_mat = bone.parent.matrix.to_3x3() if bone.parent else mathutils.Matrix.Identity(3)
    bone_rest = bone.bone.matrix_local.to_3x3()
    target_local = parent_mat.inverted() @ target_matrix
    rest_local = (bone.parent.bone.matrix_local.to_3x3().inverted() @ bone_rest) if bone.parent else bone_rest
    return (rest_local.inverted() @ target_local).to_quaternion()

# ============================================
# MAIN HEADLESS EXECUTION
# ============================================

def run_headless():
    args = get_args()
    if len(args) < 3: return
    json_path, avatar_path, output_path = args[0], args[1], args[2]

    # 1. Setup Scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.import_scene.gltf(filepath=avatar_path)
    
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if not armature:
        armature = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)

    # 2. Load Data
    data = json.load(open(json_path))
    frames = data['frames']
    fps = data['metadata']['fps']
    bpy.context.scene.render.fps = int(fps)
    
    # 3. Animate Loop (Exact copy of your Logic)
    smoother = SmartSmoother()
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    def get_bone(n): return armature.pose.bones.get(n)
    def get_vec(lm, uid, alpha=None):
        return smoother.update(uid, mediapipe_to_vector(lm), alpha if alpha is not None else 0.6)

    frame_count = 0
    for frame_idx, frame_data in enumerate(frames):
        if frame_idx % FRAME_SKIP != 0: continue
        current_frame = START_FRAME + frame_count
        hands_data = frame_data.get('hands', {})
        pose_landmarks = frame_data.get('pose')

        if pose_landmarks:
            for side, ids, pref in [('left', [11,13,15], 'Left'), ('right', [12,14,16], 'Right')]:
                p_sh, p_el, p_wr = get_vec(pose_landmarks[ids[0]], f"{side}_sh", 0.5), get_vec(pose_landmarks[ids[1]], f"{side}_el", 0.5), get_vec(pose_landmarks[ids[2]], f"{side}_wr", 0.5)
                b_arm, b_fore = get_bone(f'{pref}Arm'), get_bone(f'{pref}ForeArm')
                if b_arm:
                    b_arm.rotation_quaternion = smoother.rotation_update(f"{side}_arm_rot", solve_ball_joint(b_arm, p_sh, p_el), 0.7)
                    b_arm.keyframe_insert("rotation_quaternion", frame=current_frame)
                if b_fore:
                    b_fore.rotation_quaternion = smoother.rotation_update(f"{side}_fore_rot", solve_ball_joint(b_fore, p_el, p_wr), 0.7)
                    b_fore.keyframe_insert("rotation_quaternion", frame=current_frame)

        configs = [('left', 'Left', True), ('right', 'Right', False)]
        for side, prefix, is_left in configs:
            lms = hands_data.get(side)
            wrist_bone = get_bone(f'{prefix}Hand')
            if lms and wrist_bone:
                v_w, v_i, v_p, v_m = get_vec(lms[0], f"{side}_0", BASE_SMOOTHING), get_vec(lms[5], f"{side}_5", BASE_SMOOTHING), get_vec(lms[17], f"{side}_17", BASE_SMOOTHING), get_vec(lms[12], f"{side}_12", BASE_SMOOTHING)
                overall_closed = get_hand_closed_factor(v_w, v_m)
                wrist_bone.rotation_quaternion = smoother.rotation_update(f"{side}_wrist_r", solve_hand_rotation(v_w, v_i, v_p, wrist_bone, is_left), 0.8)
                wrist_bone.keyframe_insert("rotation_quaternion", frame=current_frame)
                
                bpy.context.view_layer.update() # CRITICAL: Update before finger solve
                
                curl_factors = {}
                for finger in FINGERS:
                    b_idx, t_idx = {'Thumb':(1,4),'Index':(5,8),'Middle':(9,12),'Ring':(13,16),'Pinky':(17,20)}[finger]
                    curl_factors[finger] = get_finger_curl_factor(get_vec(lms[b_idx], f"{side}_{b_idx}", BASE_SMOOTHING), get_vec(lms[t_idx], f"{side}_{t_idx}", FINGER_TIP_SMOOTHING), v_w)
                
                for depth in range(1, 5):
                    for fname, (start, end) in FINGER_MAPPING.items():
                        if fname.endswith(str(depth)):
                            bone = get_bone(f"{prefix}Hand{fname}")
                            if bone:
                                f_type = next(f for f in FINGERS if f in fname)
                                f_rot = solve_tracked_finger_segment(bone, get_vec(lms[start], f"{side}_{start}", THUMB_SMOOTHING if "Thumb" in fname else BASE_SMOOTHING), get_vec(lms[end], f"{side}_{end}", THUMB_SMOOTHING if "Thumb" in fname else BASE_SMOOTHING), curl_factors[f_type], depth, "Thumb" in fname, "Pinky" in fname)
                                bone.rotation_quaternion = smoother.rotation_update(f"{side}_{fname}_r", f_rot, 0.8)
                                bone.keyframe_insert("rotation_quaternion", frame=current_frame)
                    bpy.context.view_layer.update() # CRITICAL: Update after each finger segment depth
            elif wrist_bone:
                wrist_bone.rotation_quaternion = mathutils.Quaternion((1, 0, 0, 0))
                wrist_bone.keyframe_insert("rotation_quaternion", frame=current_frame)
        frame_count += 1

    # 4. Export
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_animations=True)

if __name__ == "__main__":
    run_headless()