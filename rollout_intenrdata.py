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
import pandas as pd
import sapien.core as sapien
import torch
import cv2
from scipy.spatial.transform import Rotation as R

# ===== 自前ライブラリ／ローカルパッケージ =====
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.policies.gr00t.geometry import quat2mat, mat2euler

# ===== GR00T ライブラリ =====
import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
GR00T_CHECKPOINT = Path("./checkpoints/submit/fractal_checkpoint-200000")
OUTPUT_VIDEO_DIR = Path("./mov")
DATA_CONFIG = EmbodimentTag = "google_robot" # "google_robot" or "widowx"
DEFAULT_FPS = 10
DEFAULT_CODEC = "h264"

# ---------------------------------------------------------------------------
# GR00T 推論ラッパークラス
# ---------------------------------------------------------------------------
class GR00TInference:
    """GR00T 推論用ラッパークラス"""
    def __init__(self, checkpoint_path: str, policy_setup: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        data_config = DATA_CONFIG_MAP[DATA_CONFIG]
        self.policy = Gr00tPolicy(
            model_path=str(GR00T_CHECKPOINT),
            embodiment_tag=EmbodimentTag,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            device=self.device,
        )
        print("GR00tPolicy ready.")
        self.instruction = ""
        self.use_chunk_execution = True        # False にすれば従来 (毎ステップ再推論)
        self.replan_every = None               # 例: 3 にすると 3 ステップごとに再推論 (RHC)。None ならチャンク消費し切るまで再推論しない
        self._action_chunk: np.ndarray | None = None  # shape=(T, action_dim)
        self._chunk_len: int = 0
        self._chunk_idx: int = 0
        self._last_raw_pred: dict | None = None
        self._steps_in_current_chunk: int = 0
        self._chunk_id: int = 0
        self._force_replan: bool = False
        self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        
    def reset(self, instruction: str):
        self.instruction = instruction
        self._action_chunk = None
        self._chunk_idx = 0
        self._chunk_len = 0
        self._last_raw_pred = None
        self._steps_in_current_chunk = 0
        self._chunk_id = 0
        self._force_replan = False
        self._demo_idx = 0

    def force_replan(self):
        """次ステップで強制再推論"""
        self._force_replan = True

    def _need_new_chunk(self) -> bool:
        if not self.use_chunk_execution:
            return True
        if self._action_chunk is None:
            return True
        if self._chunk_idx >= self._chunk_len:
            return True
        if self._force_replan:
            return True
        if (self.replan_every is not None and
            self.replan_every > 0 and
            self._steps_in_current_chunk >= self.replan_every):
            return True
        return False

    def step(self, image: np.ndarray, obs_full: dict | None = None):
        """1ステップ推論: obs_full から直接 8次元 state.world_vec を構築"""
        if DATA_CONFIG == "google_robot":
            resize_dim = (320, 256)  # (W,H)
        elif DATA_CONFIG == "widowx":
            resize_dim = (256, 256) # (W,H)
        else:
            raise ValueError(f"未知の DATA_CONFIG: {DATA_CONFIG}")
        
        # 画像リサイズ
        resized_image = cv2.resize(image, resize_dim)
        video_array = resized_image[None, ...]  # (1,H,W,3)

        state_vec = np.zeros((1, 7), dtype=np.float32)
        if obs_full is not None:
            tcp_pose = obs_full.get("extra", {}).get("tcp_pose", None)
            if tcp_pose is not None and len(tcp_pose) >= 7:
                tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
                state_vec[0, 0:3] = tcp_pose[:3]       # 位置(x,y,z)
                # print(state_vec)

                # クオータニオン (w,x,y,z)→(x,y,z,w) 変換
                # quat = np.roll(tcp_pose[3:7], -1)
                quat_orig = np.asarray(tcp_pose[3:7], dtype=np.float32)      # クォータニオン(qx,qy,qz,qw)
                quat = quat_orig[[0,1,2,3]]

                # クォータニオン -> オイラー角変換（Simpler_envの真似）
                # rm_bridge = quat2mat(quat)
                # rpy = mat2euler(rm_bridge @ self.default_rot.T)

                # # クォータニオン -> オイラー角変換（俺俺変換だけどgoogle_robotで確認済み）
                norm = np.linalg.norm(quat)
                if norm > 1e-12:
                    quat = quat / norm
                else:
                    quat = np.array([0,0,0,1], dtype=np.float32)

                rot = R.from_quat(quat)
                rpy = rot.as_euler('xyz', degrees=False) # roll, pitch, yaw

                print(f"quat: {quat}, rpy: {rpy}")
                state_vec[0, 3:6] = rpy  # 回転(r,p,y)
                # print(state_vec)
            try:
                tgt = np.asarray(
                    obs_full["agent"]["controller"]["gripper"]["target_qpos"],
                    dtype=np.float32,
                )
                # print(f"Gripper target qpos: {tgt}")
                bin_g = tgt[0] / 1.30999994 # 謎の正規化が必要
                state_vec[0, 6] = bin_g
            except Exception:
                pass

        # print(f"state_vec: {state_vec}")
        obs = {
            "video.image": video_array,
            "state.ee_pos": state_vec[:, 0:3],
            "state.ee_rot": state_vec[:, 3:6],
            "state.gripper": state_vec[:, 6:7],
            "annotation.human.action.task_description": [self.instruction],
        }

        # 現在ステップの state 表示
        print(f"state.ee_pos: {obs['state.ee_pos']}")
        print(f"state.ee_rot: {obs['state.ee_rot']}")
        print(f"state.gripper: {obs['state.gripper']}")

        with torch.no_grad():
            pred = self.policy.get_action(obs)

        action = {
            "delta_ee_pos": np.zeros(3, dtype=np.float32),
            "delta_ee_rot": np.zeros(3, dtype=np.float32),
            "gripper": np.zeros(1, dtype=np.float32),
        }
        
        action_pos = np.asarray(pred["action.delta_ee_pos"], dtype=np.float32)  # (T,3)
        action_rot = np.asarray(pred["action.delta_ee_rot"], dtype=np.float32)  # (T,3)
        action_grip = np.asarray(pred["action.gripper"], dtype=np.float32) # (T,1) or (T,)

        action["delta_ee_pos"] = action_pos[0, :3]
        action["delta_ee_rot"] = action_rot[0, :3]
        
        # グリッパ値抽出 (T,) / (T,1) 両対応
        if action_grip.ndim == 1:
            g_raw = float(action_grip[0])
        else:
            g_raw = float(action_grip[0, 0])

        # 二値化 + 反転 ( >0.5 を 1.0, それ以外を 0.0 )
        action["gripper"][0] = 0.0 if g_raw > 0.5 else 1.0
        print(f"Output: {action}")
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
    print("Instruction:", instruction)
    model.reset(instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)

    frames = [image]
    timestep = 0
    predicted_terminated = truncated = False
    success = False
    while not (truncated or success) and timestep < max_steps:
        raw_action, action = model.step(image, obs_full=obs)
        obs, _, success, truncated, info = env.step(np.concatenate([action["delta_ee_pos"], action["delta_ee_rot"], action["gripper"]]))
        # print(f"t={timestep} | obs.pos={obs['extra']['tcp_pose'][:3]}, obs.quat={obs['extra']['tcp_pose'][3:7]}, obs.gripper={obs['agent']['controller']['gripper']['target_qpos']}")
        print(f"t={timestep} | action={action}, success={success}, truncated={truncated}, info={info}")
        # print("Reset info:", obs)
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
    video_path = OUTPUT_VIDEO_DIR / f"{task_name}_{model_name}.mp4"
    run_episode(env, model, video_path)

if __name__ == "__main__":
    main()