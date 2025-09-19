import random
from typing import Any, Dict, List, Tuple

import numpy as np

from ..policies.base import AiroaBasePolicy
from .config import ManiSkill2Config
from .maniskill2_evaluator import maniskill2_evaluator


def calculate_robust_score(results: List[List[bool]], penalty_factor: float = 0.5) -> float:
    if not results:
        return 0.0
    success_rates = [np.mean(run) for run in results if run]
    if not success_rates:
        return 0.0
    mean_success_rate = np.mean(success_rates)
    std_dev = np.std(success_rates)
    robust_score = mean_success_rate - penalty_factor * std_dev
    return max(0.0, robust_score)


def pick_object_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- pick_object_visual_matching ---")
    results: List[List[bool]] = []

    direction_orientationions_arr = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]
    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 8],
        obj_init_y_range=[-0.02, 0.42, 8],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 1],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        task_name="fractal_pick_object_visual_matching",
    )

    for i in range(num_trials):
        urdf = random.choice(urdf_versions)
        orientation = random.choice(direction_orientationions_arr)
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
            additional_env_build_kwargs={**orientation, "urdf_version": urdf},
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def pick_object_variant_agg(env_policy: "AiroaBasePolicy", ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    """
    ランダムにシーン/姿勢/環境/照明を選んで評価を繰り返す。
    戻り値は各トライアルの _run_single_evaluation の結果のリスト。
    """
    print("\n--- pick_object_variant_agg ---")

    results: List[List[bool]] = []

    scenes = [
        "google_pick_coke_can_1_v4",
        "google_pick_coke_can_1_v4_alt_background",
        "google_pick_coke_can_1_v4_alt_background_2",
        "Baked_sc1_staging_objaverse_cabinet1_h870",
        "Baked_sc1_staging_objaverse_cabinet2_h870",
    ]

    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]

    envs = [
        "GraspSingleRandomObjectInScene-v0",
        "GraspSingleRandomObjectAltGoogleCameraInScene-v0",
        "GraspSingleRandomObjectAltGoogleCamera2InScene-v0",
    ]

    lightings = [None, "darker", "brighter"]

    base_kwargs: Dict[str, Any] = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 1],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        task_name="fractal_pick_object_variant_agg",
    )

    for i in range(num_trials):
        scene_name = random.choice(scenes)
        env_name = random.choice(envs)
        orientation = random.choice(object_orientation)
        lighting = random.choice(lightings)

        add_kwargs = dict(orientation)
        if lighting == "darker":
            add_kwargs["slightly_darker_lighting"] = True
        elif lighting == "brighter":
            add_kwargs["slightly_brighter_lighting"] = True

        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name=env_name,
            scene_name=scene_name,
            additional_env_build_kwargs=add_kwargs,
            episode_id=i,
        )

        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def pick_object_among_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- pick_object_among_visual_matching ---")
    results: List[List[bool]] = []

    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]
    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 1],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        task_name="fractal_pick_object_among_visual_matching",
    )

    for i in range(num_trials):
        urdf = random.choice(urdf_versions)
        orientation = random.choice(object_orientation)
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="GraspSingleRandomObjectDistractorInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
            additional_env_build_kwargs={**orientation, "urdf_version": urdf, "distractor_config": "less"},
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def pick_object_among_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- pick_object_among_variant_agg ---")
    results: List[List[bool]] = []

    scenes = [
        "google_pick_coke_can_1_v4",
        "google_pick_coke_can_1_v4_alt_background",
        "google_pick_coke_can_1_v4_alt_background_2",
        "Baked_sc1_staging_objaverse_cabinet1_h870",
        "Baked_sc1_staging_objaverse_cabinet2_h870",
    ]
    envs = [
        "GraspSingleRandomObjectDistractorInScene-v0",
        "GraspSingleRandomObjectDistractorAltGoogleCameraInScene-v0",
        "GraspSingleRandomObjectDistractorAltGoogleCamera2InScene-v0",
    ]
    object_orientation = [
        {"lr_switch": True},
        {"upright": True},
        {"laid_vertically": True},
    ]
    lightings = [None, "darker", "brighter"]

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.20, 0.20, 1],
        obj_init_x_range=[-0.35, -0.12, 5],
        obj_init_y_range=[-0.02, 0.42, 5],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 1],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        task_name="fractal_pick_object_among_variant_agg",
    )

    for i in range(num_trials):
        scene = random.choice(scenes)
        env = random.choice(envs)
        orientation = random.choice(object_orientation)
        lighting = random.choice(lightings)

        add_kwargs = {**orientation, "distractor_config": "less"}
        if lighting == "darker":
            add_kwargs["slightly_darker_lighting"] = True
        elif lighting == "brighter":
            add_kwargs["slightly_brighter_lighting"] = True

        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name=env,
            scene_name=scene,
            additional_env_build_kwargs=add_kwargs,
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def drawer_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- drawer_visual_matching ---")
    results: List[List[bool]] = []

    env_names = [
        "OpenTopDrawerCustomInScene-v0",
        "OpenMiddleDrawerCustomInScene-v0",
        "OpenBottomDrawerCustomInScene-v0",
        "CloseTopDrawerCustomInScene-v0",
        "CloseMiddleDrawerCustomInScene-v0",
        "CloseBottomDrawerCustomInScene-v0",
    ]
    urdf_versions = ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]

    base = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=113,
        ckpt_path=ckpt_path,
        robot_init_rot_quat_center=[0, 0, 0, 1],
        obj_init_x_range=[0, 0, 1],
        obj_init_y_range=[0, 0, 1],
        scene_name="dummy_drawer",
        task_name="fractal_drawer_visual_matching",
    )

    overlay_poses = [
        dict(
            robot_init_x_range=[0.644, 0.644, 1],
            robot_init_y_range=[-0.179, -0.179, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png",
        ),
        dict(
            robot_init_x_range=[0.765, 0.765, 1],
            robot_init_y_range=[-0.182, -0.182, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.02, -0.02, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png",
        ),
        dict(
            robot_init_x_range=[0.889, 0.889, 1],
            robot_init_y_range=[-0.203, -0.203, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.06, -0.06, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png",
        ),
        dict(
            robot_init_x_range=[0.652, 0.652, 1],
            robot_init_y_range=[0.009, 0.009, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png",
        ),
        dict(
            robot_init_x_range=[0.752, 0.752, 1],
            robot_init_y_range=[0.009, 0.009, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png",
        ),
        dict(
            robot_init_x_range=[0.851, 0.851, 1],
            robot_init_y_range=[0.035, 0.035, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png",
        ),
        dict(
            robot_init_x_range=[0.665, 0.665, 1],
            robot_init_y_range=[0.224, 0.224, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png",
        ),
        dict(
            robot_init_x_range=[0.765, 0.765, 1],
            robot_init_y_range=[0.222, 0.222, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.025, -0.025, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png",
        ),
        dict(
            robot_init_x_range=[0.865, 0.865, 1],
            robot_init_y_range=[0.222, 0.222, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.025, -0.025, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png",
        ),
    ]

    add_base = dict(station_name="mk_station_recolor", light_mode="simple", disable_bad_material=True)

    for i in range(num_trials):
        urdf = random.choice(urdf_versions)
        env_name = random.choice(env_names)
        pose = random.choice(overlay_poses)
        cfg = ManiSkill2Config(
            **base,
            env_name=env_name,
            additional_env_build_kwargs={**add_base, "urdf_version": urdf},
            **pose,
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def drawer_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- drawer_variant_agg ---")
    results: List[List[bool]] = []

    env_names = [
        "OpenTopDrawerCustomInScene-v0",
        "OpenMiddleDrawerCustomInScene-v0",
        "OpenBottomDrawerCustomInScene-v0",
        "CloseTopDrawerCustomInScene-v0",
        "CloseMiddleDrawerCustomInScene-v0",
        "CloseBottomDrawerCustomInScene-v0",
    ]

    base = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=113,
        ckpt_path=ckpt_path,
        robot_init_x_range=[0.65, 0.85, 3],
        robot_init_y_range=[-0.2, 0.2, 3],
        robot_variation_mode="episode_xy",
        robot_episode_range=[0, 1],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
        obj_init_x_range=[0, 0, 1],
        obj_init_y_range=[0, 0, 1],
        task_name="fractal_drawer_variant_agg",
    )

    background_scenes = ["frl_apartment_stage_simple", "modern_bedroom_no_roof", "modern_office_no_roof"]
    stations = ["mk_station", "mk_station2", "mk_station3"]
    lightings = [None, "brighter", "darker"]

    for i in range(num_trials):
        env_name = random.choice(env_names)
        scene = random.choice(background_scenes)
        station = random.choice(stations)
        # enable_raytracing = random.choice([True, False])
        light = random.choice(lightings)

        additional_env_build_kwargs = {
            # "shader_dir": "rt", # v100とかはray tracingに対応していない
            "light_mode": light,
            "station_name": station,
        }

        cfg = ManiSkill2Config(
            **base,
            env_name=env_name,
            scene_name=scene,
            additional_env_build_kwargs=additional_env_build_kwargs,
            enable_raytracing=False,
            episode_id=i,
        )

        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def move_near_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- move_near_visual_matching ---")
    results: List[List[bool]] = []

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=160,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.21, 0.21, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 1],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        ckpt_path=ckpt_path,
        task_name="fractal_move_near_visual_matching",
    )

    urdf_versions = [None, "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]

    for i in range(num_trials):
        urdf = random.choice(urdf_versions)
        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name="MoveNearGoogleBakedTexInScene-v0",
            scene_name="google_pick_coke_can_1_v4",
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png",
            additional_env_save_tags="baked_except_bpb_orange",
            additional_env_build_kwargs={"urdf_version": urdf},
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def move_near_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- move_near_variant_agg ---")
    results: List[List[bool]] = []

    base_kwargs = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=80,
        robot_init_x_range=[0.35, 0.35, 1],
        robot_init_y_range=[0.21, 0.21, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 1],  # TODO: widowxとおなじようランダマイズする
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.09, -0.09, 1],
        task_name="fractal_move_near_variant_agg",
        ckpt_path=ckpt_path,
    )

    envs = ["MoveNearGoogleInScene-v0", "MoveNearAltGoogleCameraInScene-v0", "MoveNearAltGoogleCamera2InScene-v0"]
    scenes = [
        "google_pick_coke_can_1_v4",
        "google_pick_coke_can_1_v4_alt_background",
        "google_pick_coke_can_1_v4_alt_background_2",
        "Baked_sc1_staging_objaverse_cabinet1_h870",
        "Baked_sc1_staging_objaverse_cabinet2_h870",
    ]
    extras = [None, {"no_distractor": True}, {"slightly_darker_lighting": True}, {"slightly_brighter_lighting": True}]

    for i in range(num_trials):
        env = random.choice(envs)
        scene = random.choice(scenes)
        extra = random.choice(extras)

        cfg = ManiSkill2Config(
            **base_kwargs,
            env_name=env,
            scene_name=scene,
            additional_env_build_kwargs=extra if extra else None,
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def put_in_drawer_visual_matching(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- put_in_drawer_visual_matching ---")
    results: List[List[bool]] = []

    env_names = ["PlaceIntoClosedTopDrawerCustomInScene-v0"]
    urdf_versions = ["recolor_cabinet_visual_matching_1", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", None]

    base = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=400,
        ckpt_path=ckpt_path,
        robot_init_rot_quat_center=[0, 0, 0, 1],
        obj_init_x_range=[-0.08, -0.02, 3],
        obj_init_y_range=[-0.02, 0.08, 3],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 1],
        task_name="fractal_put_in_drawer_visual_matching",
    )

    overlay_poses = [
        dict(
            robot_init_x_range=[0.644, 0.644, 1],
            robot_init_y_range=[-0.179, -0.179, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, -0.03, -0.03, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png",
        ),
        dict(
            robot_init_x_range=[0.652, 0.652, 1],
            robot_init_y_range=[0.009, 0.009, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png",
        ),
        dict(
            robot_init_x_range=[0.665, 0.665, 1],
            robot_init_y_range=[0.224, 0.224, 1],
            robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
            rgb_overlay_path="./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png",
        ),
    ]

    model_ids = [
        "baked_opened_pepsi_can_v2",
        "baked_opened_coke_can_v2",
        "baked_opened_7up_can_v2",
        "baked_opened_redbull_can_v2",
        "baked_blue_plastic_bottle_v2",
        "baked_apple_v2",
        "baked_orange_v2",
        "baked_sponge_v2",
    ]

    for i in range(num_trials):
        env_name = random.choice(env_names)
        urdf = random.choice(urdf_versions)
        pose = random.choice(overlay_poses)
        model_id = random.choice(model_ids)

        additional_env_build_kwargs = {
            "station_name": "mk_station_recolor",
            "light_mode": "simple",
            "disable_bad_material": True,
            "model_ids": model_id,
            "urdf_version": urdf,
        }

        cfg = ManiSkill2Config(
            **base,
            env_name=env_name,
            scene_name="dummy_drawer",
            additional_env_build_kwargs=additional_env_build_kwargs,
            **pose,
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


def put_in_drawer_variant_agg(env_policy: AiroaBasePolicy, ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
    print("\n--- put_in_drawer_variant_agg ---")
    results: List[List[bool]] = []

    common = dict(
        robot="google_robot_static",
        policy_setup="google_robot",
        control_freq=3,
        sim_freq=513,
        max_episode_steps=400,
        ckpt_path=ckpt_path,
        robot_init_rot_quat_center=[0, 0, 0, 1],
        obj_init_x_range=[-0.08, -0.02, 3],
        obj_init_y_range=[-0.02, 0.08, 3],
        obj_variation_mode="episode_xy",
        obj_episode_range=[0, 1],
        robot_init_x_range=[0.65, 0.65, 1],
        robot_init_y_range=[-0.2, 0.2, 3],
        robot_variation_mode="episode_xy",
        robot_episode_range=[0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0.0, 0.0, 1],
        task_name="fractal_put_in_drawer_variant_agg",
    )

    env_names = ["PlaceIntoClosedTopDrawerCustomInScene-v0"]
    background_scenes = ["modern_bedroom_no_roof", "modern_office_no_roof"]
    stations = ["mk_station", "mk_station2", "mk_station3"]
    lightings = [None, "brighter", "darker"]
    model_ids = [
        "pepsi_can",
        "baked_opened_pepsi_can_v2",
        "coke_can",
        "opened_coke_can",
        "baked_opened_coke_can_v2",
        "sprite_can",
        "opened_sprite_can",
        "baked_opened_7up_can_v2",
        "fanta_can",
        "opened_fanta_can",
        "redbull_can",
        "baked_opened_redbull_can_v2",
        "blue_plastic_bottle",
        "baked_blue_plastic_bottle_v2",
        "apple",
        "baked_apple_v2",
        "orange",
        "baked_orange_v2",
        "sponge",
        "baked_sponge_v2",
    ]

    for i in range(num_trials):
        env_name = random.choice(env_names)
        scene = random.choice(background_scenes)
        station = random.choice(stations)
        light = random.choice(lightings)
        model_id = random.choice(model_ids)

        additional_env_build_kwargs = {
            "model_ids": model_id,
            # "shader_dir": "rt",
            "light_mode": light,
            "station_name": station,
        }

        cfg = ManiSkill2Config(
            **common,
            env_name=env_name,
            scene_name=scene,
            additional_env_build_kwargs=additional_env_build_kwargs,
            episode_id=i,
        )
        results.append(maniskill2_evaluator(env_policy, cfg))

    return results


# ======================================================================
# 総合評価（重み付け・スコアリングは従来どおり）
# ======================================================================
SIM_WEIGHT = 0.4
VISUAL_MATCHING_WEIGHT = 0.6


def run_comprehensive_evaluation(env_policy: AiroaBasePolicy, ckpt_path: str) -> Dict[str, float]:
    print("=" * 80)
    print(f"🚀 STARTING COMPREHENSIVE EVALUATION 🚀")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Weights: Sim={SIM_WEIGHT}, VisualMatching={VISUAL_MATCHING_WEIGHT}")
    print("=" * 80)

    # fix seed
    random.seed(42)
    np.random.seed(42)

    vm_results: List[List[bool]] = []
    sim_results: List[List[bool]] = []
    num_trials = 3

    vm_results += pick_object_visual_matching(env_policy, ckpt_path, num_trials)
    sim_results += pick_object_variant_agg(env_policy, ckpt_path, num_trials)

    vm_results += pick_object_among_visual_matching(env_policy, ckpt_path, num_trials)
    sim_results += pick_object_among_variant_agg(env_policy, ckpt_path, num_trials)

    vm_results += drawer_visual_matching(env_policy, ckpt_path, num_trials)
    sim_results += drawer_variant_agg(env_policy, ckpt_path, num_trials)

    vm_results += move_near_visual_matching(env_policy, ckpt_path, num_trials)
    sim_results += move_near_variant_agg(env_policy, ckpt_path, num_trials)

    vm_results += put_in_drawer_visual_matching(env_policy, ckpt_path, num_trials)
    sim_results += put_in_drawer_variant_agg(env_policy, ckpt_path, num_trials)

    # ロバストスコア
    sim_score = calculate_robust_score(sim_results)
    vm_score = calculate_robust_score(vm_results)

    total_weight = SIM_WEIGHT + VISUAL_MATCHING_WEIGHT
    final_score = 0.0 if total_weight == 0 else (sim_score * SIM_WEIGHT + vm_score * VISUAL_MATCHING_WEIGHT) / total_weight

    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY 📊")
    print("-" * 80)
    print(f"Simulation Score (Robust):            {sim_score:.4f}")
    print(f"  - Total Simulation Runs: {len(sim_results)}")
    print(f"Visual Matching Score (Robust):       {vm_score:.4f}")
    print(f"  - Total Visual Matching Runs: {len(vm_results)}")
    print("-" * 80)
    print(f"🏆 Final Weighted Score:               {final_score:.4f}")
    print("=" * 80)

    return {
        "final_score": final_score,
        "simulation_robust_score": sim_score,
        "visual_matching_robust_score": vm_score,
    }


# def pick_object_variant_agg(env_policy: "AiroaBasePolicy", ckpt_path: str, num_trials: int = 30) -> List[List[bool]]:
#     print("\n--- pick_object_variant_agg (single-pool, one-pick-per-trial) ---")

#     results: List[List[bool]] = []

#     scenes = [
#         "google_pick_coke_can_1_v4",
#         "google_pick_coke_can_1_v4_alt_background",
#         "google_pick_coke_can_1_v4_alt_background_2",
#         "Baked_sc1_staging_objaverse_cabinet1_h870",
#         "Baked_sc1_staging_objaverse_cabinet2_h870",
#     ]
#     envs = [
#         "GraspSingleRandomObjectInScene-v0",
#         "GraspSingleRandomObjectAltGoogleCameraInScene-v0",
#         "GraspSingleRandomObjectAltGoogleCamera2InScene-v0",
#     ]
#     object_orientation = [
#         {"lr_switch": True},
#         {"upright": True},
#         {"laid_vertically": True},
#     ]
#     lightings = [None, "darker", "brighter"]

#     base_kwargs: Dict[str, Any] = dict(
#         robot="google_robot_static",
#         policy_setup="google_robot",
#         control_freq=3,
#         sim_freq=513,
#         max_episode_steps=160,
#         ckpt_path=ckpt_path,
#         robot_init_x_range=[0.35, 0.35, 1],
#         robot_init_y_range=[0.20, 0.20, 1],
#         obj_init_x_range=[-0.35, -0.12, 5],
#         obj_init_y_range=[-0.02, 0.42, 5],
#         obj_variation_mode="episode_xy",
#         obj_episode_range=[0, 4],
#         robot_init_rot_quat_center=[0, 0, 0, 1],
#         robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
#     )

#     # 単一プール（どれが選ばれたか識別できるよう (kind, value) で保持）
#     pool = (
#         [("scene", s) for s in scenes]
#         + [("env", e) for e in envs]
#         + [("orientation", o) for o in object_orientation]
#         + [("lighting", l) for l in lightings]
#     )

#     # 現在の設定（初期値）
#     current = {
#         "scene": scenes[0],
#         "env": envs[0],
#         "orientation": object_orientation[0],
#         "lighting": None,
#     }

#     for _ in range(num_trials):
#         kind, value = random.choice(pool)  # ← 毎回1つだけ選ぶ
#         current[kind] = value  # 選ばれた種類だけ更新

#         # additional_env_build_kwargs を構築
#         add_kwargs = dict(current["orientation"])
#         if current["lighting"] == "darker":
#             add_kwargs["slightly_darker_lighting"] = True
#         elif current["lighting"] == "brighter":
#             add_kwargs["slightly_brighter_lighting"] = True

#         cfg = ManiSkill2Config(
#             **base_kwargs,
#             env_name=current["env"],
#             scene_name=current["scene"],
#             additional_env_build_kwargs=add_kwargs,
#         )
#         results.append(maniskill2_evaluator(env_policy, cfg))

#     return results
