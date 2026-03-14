"""
greetings.py - Wave detection using RealSense camera.
When a hand wave is detected, the robot's claw waves back.
"""
import time
import cv2
import numpy as np
import mediapipe as mp
from cortano import RealsenseCamera, VexV5

# Motor indices (from robot config)
MOTOR_ARM = 7

# Wave detection settings
WAVE_COOLDOWN     = 3.0   # seconds before another wave can trigger
WAVE_MOVE_COUNT   = 3     # how many times claw goes up/down
WAVE_POWER        = 60    # motor power for waving
WAVE_STEP_TIME    = 0.35  # seconds each up/down step

# A wave is detected when the wrist x-position changes direction
# enough times within the tracking window
HISTORY_SECONDS   = 1.5
DIRECTION_CHANGES = 3     # min direction reversals to count as a wave
MOVEMENT_THRESH   = 0.03  # min x-movement fraction to count as a direction change

def wave_arm(robot):
    for _ in range(WAVE_MOVE_COUNT):
        robot.motor[MOTOR_ARM] = WAVE_POWER
        time.sleep(WAVE_STEP_TIME)
        robot.motor[MOTOR_ARM] = -WAVE_POWER
        time.sleep(WAVE_STEP_TIME)
    robot.motor[MOTOR_ARM] = 0


class WaveDetector:
    def __init__(self):
        self._history = []  # list of (timestamp, x_position)

    def update(self, x_pos):
        """Feed normalized wrist x position. Returns True if wave detected."""
        now = time.time()
        self._history.append((now, x_pos))
        # keep only recent history
        self._history = [(t, x) for t, x in self._history if now - t <= HISTORY_SECONDS]

        if len(self._history) < 6:
            return False

        # count direction changes in x movement
        positions = [x for _, x in self._history]
        changes = 0
        last_dir = 0
        for i in range(1, len(positions)):
            delta = positions[i] - positions[i - 1]
            if abs(delta) < MOVEMENT_THRESH / 10:
                continue
            direction = 1 if delta > 0 else -1
            if last_dir != 0 and direction != last_dir:
                # check the overall movement span to avoid micro-jitter
                span = max(positions) - min(positions)
                if span >= MOVEMENT_THRESH:
                    changes += 1
            last_dir = direction

        return changes >= DIRECTION_CHANGES


def main():
    print("Initializing camera and robot...")
    camera = RealsenseCamera()
    robot = VexV5()
    time.sleep(3)  # allow rx channel to sync

    sensors, battery = robot.sensors()
    print(f"Battery: {battery}%")

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    detector       = WaveDetector()
    last_wave_time = 0
    waving         = False

    print("Ready! Wave at the camera to make the robot wave back.")
    print("Press 'q' to quit.\n")

    while robot.running():
        color, depth = camera.read()
        if color is None:
            continue

        frame = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hand_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # use wrist landmark (0) x position for wave tracking
                wrist_x = hand_landmarks.landmark[0].x
                hand_detected = True

                now = time.time()
                cooldown_ok = (now - last_wave_time) > WAVE_COOLDOWN

                if cooldown_ok and not waving:
                    if detector.update(wrist_x):
                        print("Wave detected! Waving back...")
                        last_wave_time = now
                        waving = True
                        # wave in a background thread so camera keeps running
                        import threading
                        threading.Thread(target=wave_arm, args=(robot,), daemon=True).start()

        if waving and (time.time() - last_wave_time) > (WAVE_MOVE_COUNT * WAVE_STEP_TIME * 2 + 0.5):
            waving = False

        # overlay status
        h, w = frame.shape[:2]
        status = "WAVING BACK!" if waving else ("Hand detected - wave!" if hand_detected else "Waiting for wave...")
        color_text = (0, 200, 0) if waving else ((0, 255, 255) if hand_detected else (200, 200, 200))
        cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_text, 2)
        cv2.putText(frame, f"Battery: {battery}%", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.imshow("Greetings", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    robot.motor[MOTOR_ARM] = 0
    hands.close()
    cv2.destroyAllWindows()
    robot.stop()
    print("Goodbye!")


if __name__ == "__main__":
    main()
