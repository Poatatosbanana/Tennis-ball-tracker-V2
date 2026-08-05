#!/usr/bin/env python3
#
# 9-tennis
#
# Robot "moving-or-not" dance -- a simplified, demo-safe reaction.
#
# Instead of a four-way emotion classifier, this is a binary test: was the
# detected ball MOVING or NOT?
#   * average speed >= MOVING_THRESHOLD_MPS  -> happy dance
#   * average speed <  MOVING_THRESHOLD_MPS  -> sad dance
# The pose sequences come from README.md's "EMOTION MODE VALUES" section.
#
# Two ways to use this file:
#   * As a module:  import dance; dance.run_for_speed(robot, avg_speed_mps)
#   * Standalone (to test a dance on the robot without the ball pipeline):
#       python dance.py sad
#       python dance.py happy
#       python dance.py 5.0        # picks happy/sad for 5.0 m/s
#
# --- Units, and why the poses are in "degrees" ---
# The outline/README poses are written as AX-servo degrees where 150 = the
# centered/neutral position (e.g. "Motor 0: 150->90"). But robot.move() takes
# RADIANS, centered at 0. deg_to_rad() below bridges the two:
#   rad = radians(deg - 150)
# so 150 deg -> 0 rad (center), and every pose used here stays inside the
# servo's usable +/-90 deg (+/-1.57 rad) range. Motor 0 = pan, Motor 1 = tilt.
#

import sys
import time
import math

from gretchen.robot import Robot

# --- Standalone-run device settings (only used by main()) --------------------
# Windows: motor 'COM4', camera 0. Change to match your machine.
MOTOR_PORT = 'COM4'
CAMERA_INDEX = 0

# --- Pose / timing tunables --------------------------------------------------
DEG_NEUTRAL = 150       # servo degree value that means "centered"
MOVE_PAUSE_S = 0.4      # pause after each move so the servo has time to travel
REVERT_DELAY_S = 2.0    # README: after a dance, wait ~2 s, then return to center

# --- Speed tunables ----------------------------------------------------------
# Moving speed is Dynamixel control-table address 32 (a raw 0..1023 value, NOT
# rev/sec). Lower = slower/controlled. The default motor config uses 150; the
# README's "15/20 rev/sec" are not achievable AX speeds, so we express the
# "faster for happy" intent as a higher raw value instead.
ADDR_MOVING_SPEED = 32
SPEED_NORMAL = 150
SPEED_FAST = 500

# --- Moving/not-moving threshold ---------------------------------------------
# THE user-tunable knob. Average ball speed (m/s) at or above this is treated as
# "moving" (happy dance); below it is "not moving" (sad dance).
#
# TUNE ON HARDWARE. Note this pairs with speed_and_dance.py averaging over ALL
# tracked frames: idle frames pull the average toward 0, so a ball that is only
# briefly thrown inside a longer recording still averages LOW. Keep this
# threshold modest and tune it to your actual demo window/throw.
MOVING_THRESHOLD_MPS = 0.5


def deg_to_rad(deg):
    """Convert an AX-servo 'degree' pose (150 = center) to the radians that
    robot.move() expects (0 = center)."""
    return math.radians(deg - DEG_NEUTRAL)


def set_speed(robot, raw_value):
    """Set the moving speed (control-table addr 32) on both motors.

    There is no speed setter in the gretchen API, so we reuse the same
    setConfig path the robot uses at start-up. `raw_value` is a raw 0..1023
    Dynamixel speed, not rev/sec.
    """
    for motor_id in robot.motors.motor_id:
        robot.motors.setConfig(motor_id, {'Speed': (2, ADDR_MOVING_SPEED, raw_value)})


def _play(robot, poses, speed=SPEED_NORMAL, pause=MOVE_PAUSE_S):
    """Play a sequence of (pan_deg, tilt_deg) poses, then revert to center.

    Every dance shares the same ending (README line 28): hold the last pose for
    ~2 s, then return to neutral 150,150 (= move(0,0)) at normal speed.
    """
    set_speed(robot, speed)
    for pan_deg, tilt_deg in poses:
        robot.move(deg_to_rad(pan_deg), deg_to_rad(tilt_deg))
        time.sleep(pause)

    time.sleep(REVERT_DELAY_S)
    if speed != SPEED_NORMAL:
        set_speed(robot, SPEED_NORMAL)   # README: revert at the normal speed
    robot.move(0.0, 0.0)                  # back to center (150,150)
    time.sleep(pause)


# --- The two dances ----------------------------------------------------------
# Each is a list of (pan_deg, tilt_deg) waypoints (pan = motor 0, tilt = motor 1,
# 150 = neutral). A motor that "holds" simply repeats 150 in every pose.

def sad_dance(robot):
    # Motor 0: 150 -> 90, Motor 1: 150 -> 110  (a single downward slump)
    _play(robot, [(90, 110)], speed=SPEED_NORMAL)


def happy_dance(robot):
    # Motor 1: 150 -> 110 -> 150 -> 110 -> 150  (a quick double nod), faster.
    _play(robot, [(150, 110), (150, 150), (150, 110), (150, 150)], speed=SPEED_FAST)


# --- Selection ---------------------------------------------------------------
DANCES = {
    'happy': happy_dance,
    'sad': sad_dance,
}


def pick_dance(avg_speed_mps):
    """Return happy_dance if the ball was moving (avg speed >= threshold),
    otherwise sad_dance."""
    if avg_speed_mps >= MOVING_THRESHOLD_MPS:
        return happy_dance
    return sad_dance


def run_for_speed(robot, avg_speed_mps):
    """Pick happy/sad for `avg_speed_mps` and play it on `robot`."""
    chosen = pick_dance(avg_speed_mps)
    state = "moving" if chosen is happy_dance else "not moving"
    print("Average speed {:.2f} m/s (threshold {:.2f}) -> {} -> {}".format(
        avg_speed_mps, MOVING_THRESHOLD_MPS, state, chosen.__name__))
    chosen(robot)


def main():
    if len(sys.argv) != 2:
        print("usage: python dance.py <happy|sad|SPEED_MPS>")
        return

    arg = sys.argv[1].lower()

    robot = Robot(MOTOR_PORT, CAMERA_INDEX)
    robot.start_motors()   # motors only -- no camera needed for a dance
    try:
        if arg in DANCES:
            print("Playing '{}' dance".format(arg))
            DANCES[arg](robot)
        else:
            try:
                speed = float(arg)
            except ValueError:
                print("usage: python dance.py <happy|sad|SPEED_MPS>")
                return
            run_for_speed(robot, speed)
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
