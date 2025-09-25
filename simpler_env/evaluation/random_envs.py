from collections import OrderedDict
from typing import List

import sapien.core as sapien
from transforms3d.euler import euler2quat
import numpy as np

from ManiSkill2_real2sim.mani_skill2_real2sim.utils.registration import register_env
from ManiSkill2_real2sim.mani_skill2_real2sim.utils.common import random_choice
from ManiSkill2_real2sim.mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnBridgeInSceneEnv
from ManiSkill2_real2sim.mani_skill2_real2sim import ASSET_DIR

DEFAULT_OBJECTS = [
    "yellow_cube_3cm",
    "green_cube_3cm",
    "eggplant",
    "bridge_spoon_generated_modified",
    "bridge_carrot_generated_modified",
]

DEFAULT_TOPS = [
    "bridge_plate_objaverse_larger",
    "table_cloth_generated_shorter",
]


# Task 1
@register_env("GraspRandomObjectInScene-v0", max_episode_steps=60)
class GraspRandomObjectInScene(PutOnBridgeInSceneEnv):
    def __init__(self, candidate_source_names: List[str] = DEFAULT_OBJECTS, grasp_hold_seconds: float = 3.0, **kwargs):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, p1 in enumerate(grid_pos):
            for j, p2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([p1, p2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
        ]

        self._placeholder_src = "__random_src_placeholder__"
        self._user_src_pool = candidate_source_names

        self._grasp_hold_steps = int(grasp_hold_seconds * 5)  # fps = 5
        self.consecutive_grasp = 0

        super().__init__(
            source_obj_name=self._placeholder_src,
            target_obj_name="dummy_sink_target_plane",
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def _initialize_actors(self):
        super()._initialize_actors()
        if self.episode_target_obj is not None:
            self.episode_target_obj.hide_visual()
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)

        if self._user_src_pool is not None:
            src_candidates = list(self._user_src_pool)
        else:
            ban = {"sink", "dummy_sink_target_plane", "table_cloth_generated_shorter"}
            ban_kw = ["sink", "plane", "cloth", "towel", "target"]
            src_candidates = [
                k for k in self.model_db.keys()
                if (k not in ban) and all(kw not in k for kw in ban_kw)
            ]
        assert len(src_candidates) > 0, "No valid source objects to grasp."

        self._source_obj_name = random_choice(src_candidates, rng=self._episode_rng)

        self.consecutive_grasp = 0
        self._grasp_success_locked = False

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"random_source_obj_name": self._source_obj_name})
        return obs, info

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            is_src_obj_grasped=False,
            consecutive_grasp=False,
            grasp_stable=False,
        )

    def evaluate(self, **kwargs):
        is_src_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
        
        if self._grasp_success_locked:
            success = True
            consecutive_grasp = True
        else:
            if is_src_obj_grasped:
                self.consecutive_grasp += 1
            else:
                self.consecutive_grasp = 0

            consecutive_grasp = self.consecutive_grasp >= self._grasp_hold_steps
            success = consecutive_grasp

            if success:
                self._grasp_success_locked = True

        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] or is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] or consecutive_grasp
        )
        self.episode_stats["grasp_stable"] = success

        return dict(
            is_src_obj_grasped=is_src_obj_grasped,
            consecutive_grasp=consecutive_grasp,
            episode_stats=self.episode_stats,
            success=success,
        )

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        return f"pick up {src_name}"


# Task 2
@register_env("StackRandomGreenYellowCubeInScene-v0", max_episode_steps=60)
class StackRandomGreenYellowCubeInScene(PutOnBridgeInSceneEnv):
    def __init__(self, **kwargs):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_xs = [0.05, 0.1]
        half_edge_length_ys = [0.05, 0.1]
        xy_configs = []
        for (half_edge_length_x, half_edge_length_y) in zip(
            half_edge_length_xs, half_edge_length_ys
        ):
            grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
            grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
            )
            for i, p1 in enumerate(grid_pos):
                for j, p2 in enumerate(grid_pos):
                    if i != j:
                        xy_configs.append(np.array([p1, p2]))

        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]])]

        self._placeholder_src = "__random_src_placeholder__"
        self._placeholder_tgt = "__random_tgt_placeholder__"

        super().__init__(
            source_obj_name=self._placeholder_src,
            target_obj_name=self._placeholder_tgt,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        green = "green_cube_3cm"
        yellow = "yellow_cube_3cm"

        if random_choice([0, 1], rng=self._episode_rng):
            src, tgt = green, yellow
        else:
            src, tgt = yellow, green

        self._source_obj_name = src
        self._target_obj_name = tgt

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({
            "top_cube": self._source_obj_name,
            "bottom_cube": self._target_obj_name,
        })
        return obs, info

    def get_language_instruction(self, **kwargs):
        src = "green block" if "green" in self.episode_source_obj.name else "yellow block"
        tgt = "yellow block" if "yellow" in self.episode_target_obj.name else "green block"
        return f"stack the {src} on the {tgt}"


# Task 3
@register_env("PutRandomObjectOnRandomTopInScene-v0", max_episode_steps=60)
class PutRandomObjectOnRandomTopInScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        candidate_source_names: List[str] = DEFAULT_OBJECTS,
        candidate_target_names: List[str] = DEFAULT_TOPS,
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xy_configs = []
        for i, p1 in enumerate(grid_pos):
            for j, p2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append(np.array([p1, p2]))

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
        ]

        self._placeholder_src = "__random_src_placeholder__"
        self._placeholder_tgt = "__random_tgt_placeholder__"

        self._user_src_pool = candidate_source_names
        self._user_tgt_pool = candidate_target_names

        super().__init__(
            source_obj_name=self._placeholder_src,
            target_obj_name=self._placeholder_tgt,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        if self._user_src_pool is not None:
            src_candidates = list(self._user_src_pool)
        else:
            ban = {"sink", "dummy_sink_target_plane"}
            ban_kw = ["sink", "plane", "cloth", "towel", "target"]
            src_candidates = [
                k for k in self.model_db.keys()
                if (k not in ban) and all(kw not in k for kw in ban_kw)
            ]

        if self._user_tgt_pool is not None:
            tgt_candidates = list(self._user_tgt_pool)
        else:
            prefer_kw = ["plate", "bowl", "tray", "cloth", "towel"]
            tgt_candidates = [
                k for k in self.model_db.keys()
                if any(kw in k for kw in prefer_kw)
            ]

        assert len(src_candidates) > 0, "No valid source objects for random put-on task."
        assert len(tgt_candidates) > 0, "No valid container candidates for random put-on task."

        chosen_src = random_choice(src_candidates, rng=self._episode_rng)
        chosen_tgt = random_choice(tgt_candidates, rng=self._episode_rng)
        if chosen_src == chosen_tgt and len(src_candidates) > 1:
            alt = [x for x in src_candidates if x != chosen_tgt]
            chosen_src = random_choice(alt, self._episode_rng, rng=self._episode_rng)

        self._source_obj_name = chosen_src
        self._target_obj_name = chosen_tgt

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({
            "random_source_obj_name": chosen_src,
            "random_container_obj_name": chosen_tgt,
        })
        return obs, info

    def evaluate(self, **kwargs):
        tgt_name = self.episode_target_obj.name if self.episode_target_obj is not None else ""
        soft = ("cloth" in tgt_name) or ("towel" in tgt_name)
        if soft:
            return super().evaluate(success_require_src_completely_on_target=False, **kwargs)
        else:
            return super().evaluate(success_require_src_completely_on_target=True, **kwargs)

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        tgt = "plate" if "plate" in tgt_name else "towel"
        return f"put {src_name} on the {tgt}"


# Task 4
@register_env("PutRandomObjectInBasketScene-v0", max_episode_steps=120)
class PutRandomObjectInBasketScene(PutOnBridgeInSceneEnv):
    def __init__(self, candidate_source_names: List[str] = DEFAULT_OBJECTS, **kwargs):
        target_obj_name = "dummy_sink_target_plane"

        target_xy = np.array([-0.125, 0.025])
        xy_center = [-0.105, 0.206]
        half_span_x = 0.01
        half_span_y = 0.015
        num_x = 2
        num_y = 4

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1]]))
        xy_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]

        quat_configs = [
            np.array([euler2quat(0, 0, 0, "sxyz"), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, 1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, -1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
        ]

        self._placeholder_src = "__random_src_placeholder__"
        self._candidate_source_names = candidate_source_names

        super().__init__(
            source_obj_name=self._placeholder_src,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            rgb_always_overlay_objects=["sink", "dummy_sink_target_plane"],
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        self.sink_id = "sink"
        self.sink = self._build_actor_helper(
            self.sink_id,
            self._scene,
            density=self.model_db[self.sink_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.sink.name = self.sink_id

    def _initialize_actors(self):
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))
        self.sink.set_pose(
            sapien.Pose([-0.16, 0.13, 0.88], [1, 0, 0, 0])
        )
        self.sink.lock_motion()
        super()._initialize_actors()

    def evaluate(self, *args, **kwargs):
        return super().evaluate(
            success_require_src_completely_on_target=False,
            z_flag_required_offset=0.06,
            *args,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        if self._candidate_source_names is not None:
            candidates = list(self._candidate_source_names)
        else:
            forbid = {
                "sink",
                "dummy_sink_target_plane",
                "table_cloth_generated_shorter",
            }
            ban_keywords = ["sink", "plane", "cloth", "towel", "target"]
            candidates = [
                k
                for k in self.model_db.keys()
                if (k not in forbid)
                and all(kw not in k for kw in ban_keywords)
            ]

        assert len(candidates) > 0, "No valid source objects found for random basket task."
        self._source_obj_name = random_choice(candidates, rng=self._episode_rng)

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"random_source_obj_name": self._source_obj_name})
        return obs, info

    def _setup_prepackaged_env_init_config(self):
        ret = super()._setup_prepackaged_env_init_config()
        ret["robot"] = "widowx_sink_camera_setup"
        ret["scene_name"] = "bridge_table_1_v2"
        ret["rgb_overlay_path"] = str(ASSET_DIR / "real_inpainting/bridge_sink.png")
        return ret

    def _additional_prepackaged_config_reset(self, options):
        options["robot_init_options"] = {
            "init_xy": [0.127, 0.06],
            "init_rot_quat": [0, 0, 0, 1],
        }
        return False

    def _setup_lighting(self):
        if self.bg_name is not None:
            return
        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=shadow,
            scale=5,
            shadow_map_size=2048,
        )

    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        return f"put {src_name} into yellow basket"