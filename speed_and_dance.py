#!/usr/bin/env python3
#
# 9-tennis
#
# Copy of measure_ball_speed_threaded.py that reacts with a binary robot dance
# once the speed-measurement phase is over. The measurement itself is unchanged:
# one background thread captures both cameras back-to-back and detects on both,
# so a triangulated pair is always from two genuinely near-simultaneous frames;
# the main thread runs the Kalman filter's predict() at a fast, independent rate
# and only calls update() when a fresh synchronized pair is ready.
#
# What this copy adds on top of the original:
#   * It averages the ball's speed over ALL tracked frames of the run.
#   * A separate, dedicated dance robot (its own motor port) is connected UP
#     FRONT so a bad port fails fast, before the measurement.
#   * When you end the run (press a key), it plays a happy dance if that average
#     was "moving" (>= dance.MOVING_THRESHOLD_MPS) or a sad dance if "not moving"
#     (see dance.py). If the ball was never tracked, it skips the dance.
#
# The original measure_ball_speed_threaded.py is left untouched.
#
# Same camera/intrinsic/extrinsic placeholders as measure_ball_speed_threaded.py.
#
# Press a key (with a camera window focused) to end the phase and trigger the dance.
#

import time
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from example_ball_detector_basis import BallDetector
from gretchen.camera import Camera
from gretchen.robot import Robot

import dance

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

# Motor port for the DEDICATED dance robot (its own port, separate from the
# cameras). Windows: 'COM4'; Linux: '/dev/grt_motor'; Mac: '/dev/tty.usbserial-...'.
DANCE_MOTOR_PORT = 'COM4'

CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY = 100.7460, 109.1108, 289.0599, 255.4894
CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY = 50.0666, 65.3900, 477.5524, 248.3306

CAMERA_B_R = np.array([[0.9536385983038645, 0.29718534456531226, -0.04747941449341004], [-0.29867277744165915, 0.9151689588109345, -0.27066649006516313], [-0.036986427785412314, 0.27229882079161966, 0.9615016153679451]])
CAMERA_B_T = np.array([-0.18863492428698803, 0.022521342366157847, 0.04273897006824748])

ACCEL_NOISE = 50.0
MEASUREMENT_NOISE_STD = 0.165  # metres
PRINT_INTERVAL_S = 0.2

# Constant gravitational acceleration, modeled as a control input rather than
# folded into the motion model - assumes z is the vertical axis and points
# up in the calibrated world frame. Flip the sign (or move it to a different
# component) if your stereo setup's z axis doesn't point up.
GRAVITY_MPS2 = 0
GRAVITY_ACCEL = np.array([0.0, 0.0, -GRAVITY_MPS2])


def make_projection_matrix(fx, fy, cx, cy, R, t):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate(P_a, P_b, point_a, point_b):
    pts_a = np.array([[point_a[0]], [point_a[1]]], dtype=np.float64)
    pts_b = np.array([[point_b[0]], [point_b[1]]], dtype=np.float64)
    point_4d = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)
    return (point_4d[:3, 0] / point_4d[3, 0])


def plot_results(measurements, state_trajectory, uncertainties):
    if not measurements and not state_trajectory:
        print("Nothing was tracked - skipping plot.")
        return

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    if measurements:
        m = np.array(measurements)
        ax1.scatter(m[:, 0], m[:, 1], m[:, 2], c='r', marker='o', label='measurements')
    if state_trajectory:
        s = np.array(state_trajectory)
        ax1.plot(s[:, 0], s[:, 1], s[:, 2], c='b', label='Kalman state')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.legend()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(uncertainties, c='g')
    ax2.set_xlabel('sample')
    ax2.set_ylabel('position uncertainty, 1 std (m)')
    ax2.set_title('Kalman filter position uncertainty')

    plt.show()


class RateCounter:
    def __init__(self):
        self.count = 0
        self.last_check = time.time()

    def tick(self):
        self.count += 1

    def rate_and_reset(self):
        now = time.time()
        elapsed = now - self.last_check
        rate = self.count / elapsed if elapsed > 0 else 0.0
        self.count = 0
        self.last_check = now
        return rate


class CaptureDetectWorker:
    # Captures both cameras back-to-back and detects on both, every cycle,
    # so a stored pair always comes from two near-simultaneous frames.
    def __init__(self, camera_a_index, camera_b_index):
        self.camera_a = Camera(camera_a_index)
        self.camera_b = Camera(camera_b_index)
        self.detector_a = BallDetector()
        self.detector_b = BallDetector()

        self.lock = threading.Lock()
        self.latest_image_a = None
        self.latest_image_b = None
        self.latest_center_a = None
        self.latest_center_b = None
        self.latest_mask_a = None
        self.latest_mask_b = None
        self.latest_pair_id = 0  # incremented each successful capture cycle

        self.rate_counter = RateCounter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.camera_a.start()
        self.camera_b.start()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            print("Warning: capture/detect thread did not stop within timeout")

        # Release the native camera handles explicitly rather than leaving
        # that to whenever the garbage collector gets around to it - letting
        # it happen at an arbitrary later point (e.g. during plt.show()'s
        # event loop) is what caused the delayed crash after the plot opened.
        if self.camera_a.vc is not None:
            self.camera_a.vc.release()
        if self.camera_b.vc is not None:
            self.camera_b.vc.release()

    def _run(self):
        while not self.stop_event.is_set():
            ret_a, img_a, ts_a = self.camera_a.getImage()
            ret_b, img_b, ts_b = self.camera_b.getImage()

            if not ret_a or img_a is None or not ret_b or img_b is None:
                continue

            (img_a, center_a, mask_a) = self.detector_a.detect(img_a)
            (img_b, center_b, mask_b) = self.detector_b.detect(img_b)

            with self.lock:
                self.latest_image_a = img_a
                self.latest_image_b = img_b
                self.latest_center_a = center_a
                self.latest_center_b = center_b
                self.latest_mask_a = mask_a
                self.latest_mask_b = mask_b
                self.latest_pair_id += 1
            self.rate_counter.tick()

    def get_latest(self):
        with self.lock:
            return (self.latest_image_a, self.latest_image_b,
                    self.latest_center_a, self.latest_center_b,
                    self.latest_mask_a, self.latest_mask_b,
                    self.latest_pair_id)

    def get_rate(self):
        return self.rate_counter.rate_and_reset()


def make_kalman_filter():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.zeros(6)
    kf.P *= 100.0
    kf.H = np.zeros((3, 6))
    kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = 1.0
    kf.R = np.eye(3) * (MEASUREMENT_NOISE_STD ** 2)
    return kf


def set_process_model(kf, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    q = ACCEL_NOISE
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = q * dt**4 / 4
        Q[i, i + 3] = Q[i + 3, i] = q * dt**3 / 2
        Q[i + 3, i + 3] = q * dt**2

    # Control matrix for a constant acceleration input u (see GRAVITY_ACCEL):
    # position gets 0.5*dt^2*u, velocity gets dt*u.
    B = np.zeros((6, 3))
    for i in range(3):
        B[i, i] = 0.5 * dt**2
        B[i + 3, i] = dt

    kf.F = F
    kf.Q = Q
    kf.B = B


def run_dance_for_speed(dance_robot, all_speeds):
    """Speed phase is over: average the ball's speed over ALL tracked frames and
    play the matching dance (happy if moving, sad if not) on the already-connected
    dance robot. Skips the dance if the ball was never tracked. Returns the
    average speed (m/s), or None if there were no samples.

    The cameras have already been released by worker.stop(); the dance robot was
    connected up front and only its motors are running, so there is no device
    contention.
    """
    if not all_speeds:
        print("Ball was never tracked - skipping the dance.")
        return None

    avg_speed = float(np.mean(all_speeds))
    print(f"Average speed over all tracked frames: {avg_speed:.2f} m/s "
          f"({avg_speed * 3.6:.1f} km/h) over {len(all_speeds)} frame(s)")

    # A motor hiccup mid-dance must not crash the demo.
    try:
        dance.run_for_speed(dance_robot, avg_speed)
    except Exception as exc:
        print(f"Dance failed ({exc}) - continuing without crashing.")
    return avg_speed


def main():
    print("Camera intrinsics/extrinsics are still PLACEHOLDERS - replace them "
          "with calibrate_stereo.py's printed output before trusting any "
          "speed number this prints.")

    # Connect the dedicated dance robot UP FRONT so a bad port/cable fails fast,
    # before the (long) measurement rather than after it. start_motors() opens
    # only the motor serial port - it never opens a camera - so this does not
    # contend with the stereo cameras (indices 0/1).
    print(f"Connecting dance robot on {DANCE_MOTOR_PORT} ...")
    dance_robot = Robot(DANCE_MOTOR_PORT, CAMERA_A_INDEX)
    try:
        dance_robot.start_motors()
    except Exception as exc:
        print(f"Could not connect the dance robot on {DANCE_MOTOR_PORT}: {exc}")
        print("Aborting before measurement - fix the port/cable and retry.")
        return
    print("Dance robot ready.")

    worker = CaptureDetectWorker(CAMERA_A_INDEX, CAMERA_B_INDEX)
    worker.start()

    cv2.namedWindow("Camera A")
    cv2.namedWindow("Camera B")
    cv2.namedWindow("Speed")
    speed_canvas = np.zeros((150, 500, 3), dtype=np.uint8)

    P_a = make_projection_matrix(CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY,
                                  np.eye(3), np.zeros(3))
    P_b = make_projection_matrix(CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY,
                                  CAMERA_B_R, CAMERA_B_T)

    kf = make_kalman_filter()
    initialized = False
    last_time = time.time()
    last_print = 0.0
    main_rate = RateCounter()
    last_used_pair_id = 0

    measurements = []
    state_trajectory = []
    uncertainties = []
    all_speeds = []   # speed on every tracked frame (averaged at the end)

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                set_process_model(kf, dt)
                kf.predict(u=GRAVITY_ACCEL)
            main_rate.tick()

            img_a, img_b, center_a, center_b, mask_a, mask_b, pair_id = worker.get_latest()

            if img_a is not None:
                cv2.imshow("Camera A", img_a)
            if img_b is not None:
                cv2.imshow("Camera B", img_b)
            if mask_a is not None:
                cv2.imshow("Mask A", mask_a)
            if mask_b is not None:
                cv2.imshow("Mask B", mask_b)

            if (pair_id != last_used_pair_id
                    and center_a is not None and center_b is not None):
                point = triangulate(P_a, P_b, center_a, center_b)
                measurements.append(point.copy())
                if not initialized:
                    kf.x[0:3] = point
                    initialized = True
                else:
                    kf.update(point)
                last_used_pair_id = pair_id

            if initialized:
                state_trajectory.append(kf.x[0:3].copy())
                uncertainties.append(float(np.sqrt(np.trace(kf.P[0:3, 0:3]))))

                # Accumulate speed for the end-of-run average over ALL tracked
                # frames (reads filter state only; does not modify the filter).
                speed = float(np.linalg.norm(kf.x[3:6]))
                all_speeds.append(speed)

            if now - last_print > PRINT_INTERVAL_S:
                if initialized:
                    speed = float(np.linalg.norm(kf.x[3:6]))
                    speed_str = f"{speed:.2f} m/s ({speed * 3.6:.1f} km/h)"
                else:
                    speed_str = "(not initialized yet)"

                speed_canvas[:] = 0
                cv2.putText(speed_canvas, speed_str, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Speed", speed_canvas)

                print(f"loop rates (Hz) - main: {main_rate.rate_and_reset():.1f}  "
                      f"capture/detect: {worker.get_rate():.1f}")
                last_print = now

            key = cv2.waitKey(1)
            if key > 0:
                break
    finally:
        worker.stop()
        cv2.destroyAllWindows()

    # Speed phase is over - react with a dance before the (blocking) plot so the
    # robot doesn't wait for you to close the plot windows first. The dance robot
    # was connected up front; disconnect it no matter what once the dance is done.
    try:
        run_dance_for_speed(dance_robot, all_speeds)
    finally:
        dance_robot.disconnect()

    plot_results(measurements, state_trajectory, uncertainties)


if __name__ == '__main__':
    main()
