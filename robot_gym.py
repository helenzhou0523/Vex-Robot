"""
Minimal RL experiment: VexV5 robot navigates toward a can (AprilTag 29)
using its claw (AprilTag 28) as reference. State: [distance, angle].
Actions: discrete {stop, forward, turn_left, turn_right}.
Algorithm: DQN from dqn.py.

Usage:
  python robot_gym.py            # train from scratch
  python robot_gym.py --eval     # run saved policy (greedy)
"""

import argparse
import time
import numpy as np
import torch
import cv2
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import pyapriltags

from cortano import VexV5, RealsenseCamera
from dqn import DQN

# ─── Gym Environment ──────────────────────────────────────────────────────────

class RobotEnv(gym.Env):
    """
    Observation: [distance_to_can (m), angle_to_can (rad)]  shape=(2,)
    Action (discrete, 4):
        0 = stop
        1 = forward
        2 = turn left
        3 = turn right
    Episode ends when claw is within 5 cm and 0.1 rad of the can, or after
    MAX_STEPS steps. Manual reset is required between episodes (place the robot).
    """
    metadata = {"render_modes": []}
    MAX_STEPS = 100
    STEP_DUR  = 0.3  # seconds per action

    # Motor speeds (scaled to -100..100)
    ACTIONS = np.array([
        [ 0,  0],   # stop
        [50, 50],   # forward
        [-40, 40],  # turn left
        [40, -40],  # turn right
    ], dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.camera   = RealsenseCamera()
        self.robot    = VexV5()
        self.detector = pyapriltags.Detector(families="tag16h5")

        self.observation_space = spaces.Box(
            low  = np.array([0.0,   -np.pi], dtype=np.float32),
            high = np.array([5.0,    np.pi], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self._steps = 0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _observe(self):
        """Return (obs, detected) where obs is [distance, angle]."""
        color, _ = self.camera.read()
        gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        tags  = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.camera.fx, self.camera.fy,
                           self.camera.cx, self.camera.cy],
            tag_size=1.5,
        )
        claw_tag = can_tag = None
        for t in tags:
            if t.pose_err > 0.1 or t.decision_margin < 10.0:
                continue
            if t.tag_id == 28:
                claw_tag = t
            elif t.tag_id == 29:
                can_tag  = t

        if claw_tag is None or can_tag is None:
            return np.array([5.0, np.pi], dtype=np.float32), False

        T_cam_claw = np.eye(4)
        T_cam_claw[:3, :3] = claw_tag.pose_R
        T_cam_claw[:3,  3] = claw_tag.pose_t.flatten()

        T_cam_can = np.eye(4)
        T_cam_can[:3, :3] = can_tag.pose_R
        T_cam_can[:3,  3] = can_tag.pose_t.flatten()

        T = np.linalg.inv(T_cam_claw) @ T_cam_can
        dx, dz   = T[0, 3], T[2, 3]
        distance = np.sqrt(dx**2 + dz**2) * 0.0254  # inches → metres
        angle    = np.arctan2(-dx, dz)
        return np.array([distance, angle], dtype=np.float32), True

    def _set_motors(self, action_idx):
        left, right = self.ACTIONS[action_idx]
        self.robot.motor[0] = int(left)
        self.robot.motor[9] = int(right)

    def _stop(self):
        self.robot.motor[0] = 0
        self.robot.motor[9] = 0

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._stop()
        self._steps = 0
        input("Place the robot at the start position, then press Enter...")
        obs, _ = self._observe()
        return obs, {}

    def step(self, action):
        self._set_motors(int(action))
        time.sleep(self.STEP_DUR)
        self._stop()

        obs, detected = self._observe()
        self._steps  += 1

        distance, angle = obs
        reward  = -distance - abs(angle)          # shaped: closer + aligned = higher
        success = detected and distance < 0.05 and abs(angle) < 0.1
        timeout = self._steps >= self.MAX_STEPS
        terminated = bool(success)
        truncated  = bool(timeout)

        if success:
            reward += 10.0  # terminal bonus

        return obs, reward, terminated, truncated, {"detected": detected}

    def close(self):
        self._stop()


# ─── Discrete-action DQN wrapper ─────────────────────────────────────────────

def make_agent(obs_dim, all_actions, device):
    return DQN(
        state_shape  = obs_dim,
        all_actions  = all_actions,
        weights      = [64, 64],
        device       = device,
        gamma        = 0.95,
    )


# ─── Training loop ────────────────────────────────────────────────────────────

def train(episodes=200, save_path="robot_dqn.pth"):
    device = torch.device("cpu")
    env    = RobotEnv()
    obs_dim = env.observation_space.shape[0]

    # Discrete action table as float vectors for the DQN
    all_actions = RobotEnv.ACTIONS.tolist()  # list of [left, right]
    agent  = make_agent(obs_dim, all_actions, device)
    buffer = deque(maxlen=50_000)

    best_reward = -np.inf

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_loss   = 0.0
        ep_steps  = 0

        while True:
            action_vec = agent.sample(obs)            # numpy [left, right]
            # map action vector back to index for env.step
            action_idx = np.argmax(
                [np.allclose(action_vec, a, atol=1e-3) for a in RobotEnv.ACTIONS]
            )
            next_obs, reward, terminated, truncated, _ = env.step(action_idx)

            buffer.append([obs, action_vec, next_obs, reward,
                           0.0 if (terminated or truncated) else 1.0])
            loss = agent.fit(buffer, batch_size=64)

            ep_reward += reward
            ep_loss   += loss
            ep_steps  += 1
            obs = next_obs

            if terminated or truncated:
                break

        avg_loss = ep_loss / max(ep_steps, 1)
        print(f"Episode {ep:4d} | steps {ep_steps:3d} | "
              f"reward {ep_reward:7.2f} | loss {avg_loss:.4f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.state_dict(), save_path)
            print(f"  ✓ Saved best model (reward={best_reward:.2f})")

    env.close()
    print("Training done. Model saved to", save_path)


# ─── Evaluation loop ─────────────────────────────────────────────────────────

def evaluate(load_path="robot_dqn.pth", episodes=10):
    device = torch.device("cpu")
    env    = RobotEnv()
    obs_dim = env.observation_space.shape[0]
    all_actions = RobotEnv.ACTIONS.tolist()

    agent = make_agent(obs_dim, all_actions, device)
    agent.load_state_dict(torch.load(load_path, map_location=device))
    agent.steps = 100_000  # force greedy (eps ≈ 0.9)
    agent.eval()

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        while True:
            action_vec = agent.sample(obs)
            action_idx = np.argmax(
                [np.allclose(action_vec, a, atol=1e-3) for a in RobotEnv.ACTIONS]
            )
            obs, reward, terminated, truncated, info = env.step(action_idx)
            ep_reward += reward
            if terminated or truncated:
                break
        print(f"Episode {ep:3d} | reward {ep_reward:7.2f}")

    env.close()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",     action="store_true", help="run saved policy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--model",    type=str, default="robot_dqn.pth")
    args = parser.parse_args()

    if args.eval:
        evaluate(load_path=args.model, episodes=args.episodes)
    else:
        train(episodes=args.episodes, save_path=args.model)
