# bingjiez

Robotics demos for a VEX V5 robot with Intel RealSense camera, running on a Jetson device.

## Projects

### greetings.py — Wave Detection Demo

Detects hand waves via the RealSense camera using MediaPipe hand tracking. When a wave is detected, the robot's claw waves back.

### robot_gym.py — RL Navigation

Trains a DQN agent to navigate toward a target can using AprilTag markers for visual localization. The robot learns a policy to minimize distance and heading error to the target.

## Hardware Requirements

- VEX V5 Brain with RS485 communication
- Intel RealSense depth camera
- Jetson Nano/Orin
- AprilTag markers: ID 28 (robot claw), ID 29 (target can), 1.5-inch physical size

## Installation

### 1. System dependencies

```bash
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install python3-pip
sudo pip3 install --upgrade pip setuptools packaging
sudo pip3 install pyserial numpy requests opencv-python pyrealsense2
```

### 2. CortexNanoBridge

```bash
git clone https://github.com/timrobot/CortexNanoBridge.git
cd CortexNanoBridge/jetson_nano
sudo bash ./install.sh
sudo reboot
```

### 3. Python dependencies

```bash
pip3 install torch gymnasium mediapipe pyapriltags opencv-python pyrealsense2
```

## Usage

### Wave detection

```bash
python3 greetings.py
```

Wave your hand at the camera. The robot's claw will wave back. Press `q` to quit.

### RL navigation

```bash
# Train (200 episodes by default)
PYTHONPATH=/home/map/robotics:$PYTHONPATH python3 robot_gym.py

# Evaluate a trained policy
PYTHONPATH=/home/map/robotics:$PYTHONPATH python3 robot_gym.py --eval

# Custom options
PYTHONPATH=/home/map/robotics:$PYTHONPATH python3 robot_gym.py --episodes 500 --model my_model.pth
PYTHONPATH=/home/map/robotics:$PYTHONPATH python3 robot_gym.py --eval --model my_model.pth
```

## Configuration

Key parameters in `greetings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WAVE_COOLDOWN` | 3.0 s | Minimum time between wave responses |
| `WAVE_MOVE_COUNT` | 3 | Claw up/down repetitions |
| `WAVE_POWER` | 60 | Motor power (0–100) |
| `DIRECTION_CHANGES` | 3 | Reversals required to detect a wave |
| `MOVEMENT_THRESH` | 0.03 | Minimum wrist movement to register |

Key parameters in `robot_gym.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_STEPS` | 100 | Steps per episode before timeout |
| `STEP_DUR` | 0.3 s | Duration of each action |
| `gamma` | 0.95 | DQN discount factor |
| `batch_size` | 64 | Training batch size |

Actions: stop, forward, turn left, turn right.
Observation: `[distance_to_can (m), angle_to_can (rad)]`.
Reward: `-distance - |angle| + 10.0` on success.

## Dependencies

- [`cortano`](https://github.com/timrobot/CortexNanoBridge) — VEX V5 robot control and RealSense camera interface
- `torch` — DQN neural network
- `gymnasium` — RL environment interface
- `pyapriltags` — AprilTag detection
- `mediapipe` — Hand pose estimation
- `opencv-python` — Image processing
- `pyrealsense2` — RealSense camera SDK
