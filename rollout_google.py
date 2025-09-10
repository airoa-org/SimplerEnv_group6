#!/usr/bin/env python3
"""
GR00T 推論デモ（ManiSkill2 ラッパ "simpler_env" 版）
"""

# ===== 標準ライブラリ =====
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ===== サードパーティライブラリ =====
import mediapy as media
import numpy as np
import sapien.core as sapien
import torch
import cv2
from scipy.spatial.transform import Rotation as R

# ===== 自前ライブラリ／ローカルパッケージ =====
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# ===== GR00T ライブラリ =====
import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
GR00T_CHECKPOINT = Path("./checkpoints/fractal_bridge_ft_epoch10k_checkpoint-100000")
DEFAULT_VIDEO_DIR = Path("./mov")
DEFAULT_FPS = 10
DEFAULT_CODEC = "h264"

# ---------------------------------------------------------------------------
# GR00T 推論ラッパークラス
# ---------------------------------------------------------------------------
class GR00TInference:
    """GR00T 推論用ラッパークラス"""
    def __init__(self, checkpoint_path: str, policy_setup: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        data_config = DATA_CONFIG_MAP["fourier_gr1_worldvec_lite"] #fourier_gr1_worldvec_lite or widowx_bridge_lerobot
        self.policy = Gr00tPolicy(
            model_path=str(GR00T_CHECKPOINT),
            embodiment_tag="gr1",
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            device=self.device,
        )
        print("GR00tPolicy ready.")
        self.instruction = ""
        
    def reset(self, instruction: str):
        self.instruction = instruction
        
    def step(self, image: np.ndarray, obs_full: dict | None = None):
        """1ステップ推論: obs_full から直接 8次元 state.world_vec を構築"""
        resized_image = cv2.resize(image, (320, 256))
        video_array = resized_image[None, ...]  # (1,H,W,3)
        # print(f"obs_full: {obs_full}")

        state_vec = np.zeros((1, 8), dtype=np.float32)
        if obs_full is not None:
            tcp_pose = obs_full.get("extra", {}).get("tcp_pose", None)
            if tcp_pose is not None and len(tcp_pose) >= 7:
                tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
                state_vec[0, 0:3] = tcp_pose[:3]       # 位置(x,y,z)
                state_vec[0, 3:7] = tcp_pose[3:7]      # クォータニオン(qx,qy,qz,qw)
            # グリッパ2値化: 両方>=0.5 -> 1 / 両方<=0.5 -> 0 / それ以外(中間)は 0.5
            try:
                tgt = np.asarray(
                    obs_full["agent"]["controller"]["gripper"]["target_qpos"],
                    dtype=np.float32,
                )
                print(f"Gripper target qpos: {tgt}")
                bin_g = tgt[0] / 1.30999994 # 謎の正規化が必要
                state_vec[0, 7] = bin_g
            except Exception:
                pass

        print(f"state_vec: {state_vec}")
        obs = {
            "video.image": video_array,
            "state.world_vec": state_vec,
            "annotation.human.action.task_description": [self.instruction],
        }

        with torch.no_grad():
            pred = self.policy.get_action(obs)

        action = {
            "world_vector": np.zeros(3, dtype=np.float32),
            "rpy": np.zeros(3, dtype=np.float32),
            "gripper": np.zeros(1, dtype=np.float32),
        }
        if "action.world_vec" in pred:
            seq = np.asarray(pred["action.world_vec"], dtype=np.float32)  # (T,7)
            print(f"seq shape: {seq.shape}")
            first = seq[0]
            action["world_vector"] = first[0:3]
            action["rpy"] = first[3:6]
            action["gripper"] = first[6:7]

        # --- グリッパ反転（二値化） ---
        g = float(action["gripper"][0])
        action["gripper"][0] = 0.0 if g > 0.5 else 1.0
        # -----------------------------------

        return pred, action

# ---------------------------------------------------------------------------
# モデルロード関数
# ---------------------------------------------------------------------------
def load_model(model_name: str, policy_setup: str):
    if "gr00t" in model_name:
        print("GR00T を使用")
        return GR00TInference(
            checkpoint_path=str(GR00T_CHECKPOINT),
            policy_setup=policy_setup
        )
    raise ValueError(f"未知のモデル名: {model_name}")

# ---------------------------------------------------------------------------
# 環境／モデルセットアップ
# ---------------------------------------------------------------------------
def make_env(task_name: str):
    if "env" in globals():
        globals()["env"].close()
        del globals()["env"]
    env = simpler_env.make(task_name)
    sapien.render_config.rt_use_denoiser = False
    obs, reset_info = env.reset()
    return env, obs

# ---------------------------------------------------------------------------
# 推論ループ
# ---------------------------------------------------------------------------
def run_episode(
    env,
    model,
    save_path: Path,
    fps: int = DEFAULT_FPS,
    codec: str = DEFAULT_CODEC,
    max_steps: int = 300
) -> Tuple[bool, List[np.ndarray]]:
    obs, _ = env.reset()
    instruction = env.get_language_instruction()
    model.reset(instruction)
    # print("Instruction:", instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)

    frames = [image]
    timestep = 0
    predicted_terminated = truncated = False
    success = False
    while not (truncated or success) and timestep < max_steps:
        raw_action, action = model.step(image, obs_full=obs)
        obs, _, success, truncated, info = env.step(np.concatenate([action["world_vector"], action["rpy"], action["gripper"]]))
        # print(f"t={timestep} | obs.pos={obs['extra']['tcp_pose'][:3]}, obs.quat={obs['extra']['tcp_pose'][3:7]}, obs.gripper={obs['agent']['controller']['gripper']['target_qpos']}")
        print(f"t={timestep} | action={action}, success={success}, truncated={truncated}, info={info}")
        print("===========================")
        timestep += 1
        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image)

    print(f"Episode success: {success}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(save_path), np.stack(frames), fps=fps, codec=codec)
    print("Saved:", save_path)
    return success, frames

# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
def main() -> None:
    task_name = "google_robot_pick_coke_can" # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
    model_name = "gr00t"
    policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"
    env, _ = make_env(task_name)
    model = load_model(model_name, policy_setup)
    video_path = DEFAULT_VIDEO_DIR / f"{task_name}_{model_name}.mp4"
    run_episode(env, model, video_path)

if __name__ == "__main__":
    main()