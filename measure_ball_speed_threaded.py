#!/usr/bin/env python3
#
# 9-tennis
#
# Threaded version of measure_ball_speed_stationary.py: one background
# thread captures both cameras back-to-back (as close together in time as
# possible) and detects on both, so a triangulated pair is always from two
# genuinely near-simultaneous frames - not two independently-timed ones.
# The main thread runs the Kalman filter's predict() at a fast, independent
# rate and only calls update() when a fresh synchronized pair is ready.
# Also reports how fast each of the two loops (main + the capture/detect
# thread) is actually running.
#
# Same camera/intrinsic/extrinsic placeholders as measure_ball_speed_stationary.py.
#
# Press a key (with a camera window focused) to exit.
#

import time
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from example_ball_detector_basis import BallDetector
from gretchen.camera import Camera

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

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
G = 0
GRAVITY_ACCEL = np.array([0.0, 0.0, -G])


#given the place 
def make_projection_matrix(fx, fy, cx, cy, R, t):
    """Build a camera's 3x4 projection matrix P = K @ [R | t].

    Combines the intrinsics (fx, fy, cx, cy - how the camera's own lens/
    sensor maps 3D points to pixels) with its extrinsics (R, t - the
    camera's rotation/translation relative to the shared world frame, where
    camera A is the origin). cv2.triangulatePoints uses P to back-project a
    pixel into a 3D ray for this camera.

    Args:
        fx, fy: focal length in pixels.
        cx, cy: principal point in pixels.
        R: 3x3 rotation matrix, world frame -> this camera's frame.
        t: 3-element translation vector, world frame -> this camera's frame.

    Returns:
        3x4 numpy array, this camera's projection matrix.
    """
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate(P_a, P_b, point_a, point_b):
    """Reconstruct a 3D point from its 2D pixel position in both cameras.

    Args:
        P_a, P_b: projection matrices for camera A and B (see
            make_projection_matrix).
        point_a, point_b: (u, v) pixel coordinates of the detected ball
            center in camera A's and camera B's image, respectively.

    Returns:
        3-element numpy array: the triangulated (x, y, z) position in the
        world frame, in metres.
    """
    pts_a = np.array([[point_a[0]], [point_a[1]]], dtype=np.float64)
    pts_b = np.array([[point_b[0]], [point_b[1]]], dtype=np.float64)
    point_4d = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)
    return (point_4d[:3, 0] / point_4d[3, 0])


def plot_results(measurements, state_trajectory, uncertainties):
    """Show the run's results after tracking stops: a 3D plot of raw
    triangulated measurements vs. the Kalman-filtered trajectory, and a 2D
    plot of the filter's position uncertainty over time.

    Args:
        measurements: list of raw triangulated (x, y, z) points, one per
            successful stereo detection.
        state_trajectory: list of the Kalman filter's estimated (x, y, z)
            position, one per main-loop tick once initialized.
        uncertainties: list of the filter's position uncertainty (1 std,
            metres), aligned with state_trajectory.
    """
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
    """Counts how many times tick() is called per second, reset each time
    rate_and_reset() is read. Used to report the main loop's and the
    capture/detect thread's actual throughput (Hz)."""

    def __init__(self):
        """Start a fresh count with the clock running from now."""
        self.count = 0
        self.last_check = time.time()

    def tick(self):
        """Record one event (call this once per loop iteration/frame)."""
        self.count += 1

    def rate_and_reset(self):
        """Return events-per-second since the last call, then reset the
        counter and clock so the next call measures the next interval.

        Returns:
            float: events per second (0.0 if called with no elapsed time).
        """
        now = time.time()
        elapsed = now - self.last_check
        rate = self.count / elapsed if elapsed > 0 else 0.0
        self.count = 0
        self.last_check = now
        return rate


class CaptureDetectWorker:
    """Runs camera capture and ball detection on a background thread.

    Captures both cameras back-to-back and detects on both, every cycle, so
    a stored pair always comes from two near-simultaneous frames - not two
    independently-timed ones. The main thread reads the latest results via
    get_latest() whenever it wants them, without blocking on camera I/O.
    """

    def __init__(self, camera_a_index, camera_b_index):
        """Create (but do not yet open) both cameras, detectors, and the
        background thread.

        Args:
            camera_a_index, camera_b_index: OS camera device indices passed
                straight to gretchen.camera.Camera.
        """
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
        """Open both cameras and start the background capture/detect loop."""
        self.camera_a.start()
        self.camera_b.start()
        self.thread.start()

    def stop(self):
        """Signal the background thread to stop, wait for it to exit, and
        release both cameras' native handles.

        Blocks up to 2 seconds for the thread to exit cleanly; prints a
        warning (but does not raise) if it doesn't. Releasing the cameras
        here rather than leaving it to the garbage collector matters: doing
        it at an arbitrary later point (e.g. during plt.show()'s event
        loop) previously caused a delayed crash after the plot opened.
        """
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            print("Warning: capture/detect thread did not stop within timeout")

        if self.camera_a.vc is not None:
            self.camera_a.vc.release()
        if self.camera_b.vc is not None:
            self.camera_b.vc.release()

    def _run(self):
        """Background loop: capture + detect on both cameras every cycle,
        publishing the latest images/centers/masks under self.lock.

        Runs until stop_event is set. Skips a cycle entirely if either
        camera fails to deliver a frame, so latest_pair_id only advances on
        genuinely paired, near-simultaneous frames.
        """
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
        """Thread-safely snapshot the most recent capture/detect results.

        Returns:
            Tuple of (image_a, image_b, center_a, center_b, mask_a, mask_b,
            pair_id). center_a/b are (u, v) pixel coords or None if no ball
            was detected that cycle. pair_id increments once per successful
            capture cycle - compare it to a previously-seen value to tell
            whether this is a fresh pair.
        """
        with self.lock:
            return (self.latest_image_a, self.latest_image_b,
                    self.latest_center_a, self.latest_center_b,
                    self.latest_mask_a, self.latest_mask_b,
                    self.latest_pair_id)

    def get_rate(self):
        """Return the capture/detect thread's throughput (Hz) since the
        last call, then reset the counter - see RateCounter.rate_and_reset.
        """
        return self.rate_counter.rate_and_reset()


def make_kalman_filter():
    """Create a constant-velocity 3D Kalman filter for the ball's position.

    State x = [x, y, z, vx, vy, vz]. Measurements are (x, y, z) triangulated
    positions (see triangulate()); H picks out the position components
    from the state. R (measurement noise) is set from
    MEASUREMENT_NOISE_STD; F and Q are filled in per-tick by
    set_process_model() since they depend on the variable time step dt.

    Returns:
        A filterpy.kalman.KalmanFilter ready for predict()/update(), with
        an arbitrary large initial P (position/velocity both unknown at
        startup).
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.zeros(6)
    kf.P *= 100.0
    kf.H = np.zeros((3, 6))
    kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = 1.0
    kf.R = np.eye(3) * (MEASUREMENT_NOISE_STD ** 2)
    return kf


def set_process_model(kf, dt):
    """Fill in the Kalman filter's F, Q, and B for the current time step.

    Called once per main-loop tick, right before kf.predict(), since all
    three depend on dt (the time elapsed since the last predict()):
      - F: constant-velocity motion model (position += velocity * dt).
      - Q: process noise, scaled by ACCEL_NOISE - how much unmodeled
        acceleration (drag, spin, bounces, swing changes; gravity is
        handled separately via the control input, not Q) we expect between
        ticks. Larger ACCEL_NOISE lets the filter react faster to real
        motion changes at the cost of more jitter from noise; smaller
        ACCEL_NOISE smooths more but lags behind real changes.
      - B: control matrix pairing with the constant GRAVITY_ACCEL passed to
        kf.predict(u=...) - position gets 0.5*dt^2*u, velocity gets dt*u.

    Args:
        kf: the KalmanFilter to update in place.
        dt: seconds elapsed since the previous predict() call.
    """
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    q = ACCEL_NOISE
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = q * dt**4 / 4
        Q[i, i + 3] = Q[i + 3, i] = q * dt**3 / 2
        Q[i + 3, i + 3] = q * dt**2

    B = np.zeros((6, 3))
    for i in range(3):
        B[i, i] = 0.5 * dt**2
        B[i + 3, i] = dt

    kf.F = F
    kf.Q = Q
    kf.B = B


def main():
    """Entry point: run live stereo ball tracking until a key is pressed.

    Starts the background CaptureDetectWorker, then loops on the main
    thread: runs the Kalman filter's predict() every tick (fast, so the
    state stays current between measurements) and update() whenever a
    fresh stereo-triangulated pair is ready; shows the camera/mask/speed
    windows; and prints loop-rate diagnostics. On exit, stops the worker,
    closes all windows, and shows the 3D trajectory + uncertainty plots via
    plot_results().
    """
    print("Camera intrinsics/extrinsics are still PLACEHOLDERS - replace them "
          "with calibrate_stereo.py's printed output before trusting any "
          "speed number this prints.")

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

    plot_results(measurements, state_trajectory, uncertainties)


if __name__ == '__main__':
    main()

