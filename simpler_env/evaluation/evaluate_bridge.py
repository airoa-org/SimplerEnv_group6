from typing import Any, Dict, List

from ..policies.base import AiroaBasePolicy
from .config import ManiSkill2Config
from .evaluate import _run_single_evaluation, calculate_robust_score

# env_name → task_name の対応表
ENV_TO_TASK: Dict[str, str] = {
    "PutSpoonOnTableClothInScene-v0": "widowx_spoon_on_towel",
    "PutCarrotOnPlateInScene-v0": "widowx_carrot_on_plate",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "widowx_stack_cube",
    "PutEggplantInBasketScene-v0": "widowx_put_eggplant_in_basket",
}

def bridge(env_policy: AiroaBasePolicy, ckpt_path: str) -> List[List[bool]]:
    """
    scripts/bridge.sh を Python で忠実に再現。
    4 本の実行（3 + 1）を順番に回し、各 run の成功配列を返す。
    """
    print("\n--- bridge (scripts/bridge.sh) ---")
    results: List[List[bool]] = []

    # ====== bridge_table_1_v1 ======
    common_v1 = dict(
        policy_setup="widowx_bridge",
        robot="widowx",
        control_freq=5,
        sim_freq=500,
        max_episode_steps=60,
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        robot_init_x_range=[0.147, 0.147, 1],
        robot_init_y_range=[0.028, 0.028, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 24],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
    )
    envs_v1 = [
        ("PutCarrotOnPlateInScene-v0", "bridge_table_1_v1"),
        ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", "bridge_table_1_v1"),
        ("PutSpoonOnTableClothInScene-v0", "bridge_table_1_v1"),
    ]
    for env_name, scene_name in envs_v1:
        task_name = ENV_TO_TASK.get(env_name)
        cfg = ManiSkill2Config(
            **common_v1,
            env_name=env_name,
            scene_name=scene_name,
            task_name=task_name,
        )
        results.append(_run_single_evaluation(env_policy, cfg, ckpt_path))

    # ====== bridge_table_1_v2 ======
    # 注意: ここだけ max_episode_steps=120、ロボット/カメラセットアップが異なる
    cfg_v2 = ManiSkill2Config(
        policy_setup="widowx_bridge",
        robot="widowx_sink_camera_setup",
        control_freq=5,
        sim_freq=500,
        max_episode_steps=120,
        env_name="PutEggplantInBasketScene-v0",
        scene_name="bridge_table_1_v2",
        rgb_overlay_path="ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        robot_init_x_range=[0.127, 0.127, 1],
        robot_init_y_range=[0.06, 0.06, 1],
        obj_variation_mode="episode",
        obj_episode_range=[0, 24],
        robot_init_rot_quat_center=[0, 0, 0, 1],
        robot_init_rot_rpy_range=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        ckpt_path=ckpt_path,
        task_name=ENV_TO_TASK["PutEggplantInBasketScene-v0"],  # v2 はバスケット固定
    )
    results.append(_run_single_evaluation(env_policy, cfg_v2, ckpt_path))

    return results


def run_comprehensive_evaluation_bridge(env_policy: AiroaBasePolicy, ckpt_path: str) -> Dict[str, Any]:
    """
    Bridge 用の総合実行。scripts/bridge.sh を 1:1 で再現し、
    ロバストスコアも合わせて出力する（便宜機能）。
    """
    print("=" * 80)
    print("🚧 STARTING BRIDGE EVALUATION (bridge.sh) 🚧")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 80)

    runs = bridge(env_policy, ckpt_path)
    vm_score = calculate_robust_score(runs)  # overlay 相当なので Visual Matching としてロバストスコアを算出

    print("\n" + "=" * 80)
    print("📊 BRIDGE EVALUATION SUMMARY 📊")
    print("-" * 80)
    print(f"Visual Matching Score (Robust):       {vm_score:.4f}")
    print(f"  - Total Runs: {len(runs)}")
    print("=" * 80)

    return {
        "visual_matching_robust_score": vm_score,
        "num_runs": len(runs),
    }
