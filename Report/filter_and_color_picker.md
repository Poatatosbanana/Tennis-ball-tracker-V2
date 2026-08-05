# Ball Detection: Filter Improvements and the Color Picker

## Overview

My part of the project was getting the ball detection to actually work under
real, changing lighting. This came in two stages. First I tuned the
morphological filters (erode and dilate) that clean up the color mask. When
that alone was not enough, I built a small color-picker tool that re-samples
the ball's color directly from the camera, which turned out to be the change
that really made detection reliable.

## Stage 1: Tuning the erode / dilate filters

The detection pipeline is: convert the frame to HSV, threshold it with a
color range (`cv2.inRange`) to get a black-and-white mask, and then clean that
mask up with two morphological filters before looking for circular contours.

Those two filters are **erode** and **dilate**, and they run right after the
color filter:

- **Erode** shrinks the white regions. It wipes out small specks of noise that
  slipped through the color threshold, but it also eats into the edges of the
  real ball.
- **Dilate** grows the white regions back. It fills small holes and re-expands
  the ball blob so it reads as one solid shape.

The starting values were `erode(iterations=3)` followed by
`dilate(iterations=2)`. The problem was that three passes of erosion were too
aggressive: under poor lighting the ball's mask was already thin and broken up,
and eroding it three times chewed it apart before dilation could recover it. So
I changed the balance:

| Filter  | Before | After |
|---------|--------|-------|
| erode   | 3      | 2     |
| dilate  | 2      | 4     |

Dropping erosion to 2 iterations keeps more of the ball, and raising dilation to
4 iterations closes the gaps back up so the remaining blob is solid and round
enough to pass the circularity check. This is the change in
`example_ball_detector.py`.

## Stage 2: Why the filters alone weren't enough

I spent a long time playing with the color range and the filter settings trying
to find one "ideal" configuration, but the detection was still not up to our
standard. It kept dropping the ball or picking up the wrong thing.

The real cause was the lighting, not the filters. On the ball itself the
lighting is dramatic: the top of the ball (the part facing the light) and the
bottom of the ball sit at very different points in HSV. That single ball spans a
wide spread of hue, saturation, and value at the same instant.

This puts a fixed HSV range in a no-win situation:

- Make the range **too broad** to cover the whole ball, and other colors in the
  scene now fall inside it too, so you get false detections.
- Make the range **too narrow** to exclude everything else, and parts of the
  ball drop outside the range, so the mask breaks into pieces and the ball is
  not detected.

No single hand-picked range solves both at once, which is why more filter tuning
kept hitting a wall.

## Stage 3: The color picker (the budget solution)

Instead of guessing fixed HSV numbers, the color picker samples the ball's
actual color from the camera under the actual lighting we are in, and builds the
filter range from that. You left-click the ball once and it retunes the
detector.

How it works (`pick_color_from_click` in the picker script):

1. It samples a `patch` × `patch` window (default 9×9) centered on the click,
   not just the single clicked pixel.
2. It takes the **median** H, S, and V of that window. Using the median instead
   of the average makes it robust to a single hot pixel or a specular highlight
   glinting off the ball, which would otherwise drag the sample off.
3. It centers the detector's `colorLower` / `colorUpper` on that median with
   fixed margins (±12 on hue, and generous room on S and V).
4. It clamps the low end of saturation and value to floors (`s_floor=40`,
   `v_floor=80`). This is what stops a dark or black background from leaking
   back into the mask even if you happen to click a dim part of the ball.

Because the range is re-centered on whatever color the ball actually is right
now, it stays narrow enough to reject the background but is always aimed at the
correct hue. Whenever the lighting changes to a new but consistent condition,
you re-click and it adapts in a couple of seconds. There is no training and no
extra hardware, which is why this was our "budget" solution: a small tool that
solved the lighting problem that no amount of static filter tuning could.

One implementation detail worth noting: the picker converts with
`COLOR_RGB2HSV`, the same conversion the detector uses, so the color you click
lands in the exact HSV space the mask is thresholded against. Matching the
detector is what makes the picked color equal the filtered color.

## Result

The combination of a gentler erosion / stronger dilation balance and the
click-to-sample color picker gave us reliable ball detection across different
consistent lighting setups, where the fixed hand-tuned filter alone kept
failing.
