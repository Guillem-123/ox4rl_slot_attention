from yacs.config import CfgNode

moc_cfg = CfgNode({
    'adjacent_consistency_weight': 0.0,
    'pres_inconsistency_weight': 0.0,
    'area_pool_weight': 0.0,
    'area_object_weight': 10.0,
    'cosine_sim': True,
    'object_threshold': 0.5,
    'z_cos_match_weight': 5.0, # corresponds to beta + 1 in the paper
    'full_object_weight': 3000,  # Unused
    'motion_input': True,
    'motion': True,
    'motion_kind': 'mode',
    'motion_direct_weight': 1.0,  # Unused
    'motion_loss_weight_z_pres': 1000.0, #10.0
    'motion_loss_weight_z_where': 10000.0, #100.0
    'motion_loss_weight_alpha': 5, #100, #1
    'motion_weight': 100.0,
    'motion_sigmoid_steepen': 10000.0,  # Unused
    'motion_cooling_end_step': 3000,
    'motion_cooling_start_step': 0,
    'dynamic_scheduling': True,
    'agree_sim': True,
    'dynamic_steepness': 2.0,
    'use_variance': True,
    'motion_underestimating': 1.25, #2.0,
    'motion_object_found_lambda': 0.1, #0.025,
    'z_where_offset': 0.1,
    'acceptable_non_moving': 8,  # Unused
    'variance_steps': 20,
    'motion_requirement': 2.0,  # Unused
})
