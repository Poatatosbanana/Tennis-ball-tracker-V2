# Tennis ball tracker


PROJECT GOOGLE DOC: https://docs.google.com/document/d/1NunoxVIXxYMdcvpThoi8C1GibIxaJYeU--GN3O8KpII/edit?usp=sharing

PROJECT PPTX: https://docs.google.com/presentation/d/1EZ985K_s-gtmI6M6_TrzJMXqAE4Jui_3/edit?usp=sharing&ouid=110077625920651084626&rtpof=true&sd=true

## Overview
This Project is a tracking system involving two “Gretchens” to track the location and velocity of a ball in 3D space, relative to a predetermined space with assigned coordinates. This involves using a motion tracking program (where the positions of the Gretchens was declared ahead of time), a Kalman filter and a program that would calculate the deltas in the x y and z axis of a moving ball, a UI to display the recorded velocity, and a third Gretchen that uses one of four different “emotes” as a rough indicator of recorded velocity to the pitcher who may not be able to view the UI immediately after throwing.

 
## Software Requirements
Ensure "Dynawizard 2.0" and the Gretchen Development Environment (GDE) with a Python 3 Virtual Enrionment are isntalled. Ensure git is also installed so this repository can be forked and isntalled later.

## Installation

#### Python Packages

Enter the virtual environment…

```
Linux: source venv/bin/activate
Windows: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
Mac OS: source venv/bin/activate
```

… and install the libraries cv2, dlib, imutils, numpy, matplotlib, filterpy.kalman…

```
pip install pyserial setuptools opencv-contrib-python imutils dlib
pip install numpy
pip install matplotlib
pip install filterpy
```

… and import the relevent libraries into the virtual environment
```
import cv2
import imutils
import numpy
import dlib
import matplotlib
import filterpy
print("CV2: {}, imutils: {}, Dlib: {}, numpy: {}, matplotlib, filterpy: {}.format(cv2.__version__, imutils.__version__, numpy.__version__, dlib.__version__, mathplotlib.__version__, filterpy.__version))

```

## Calibration

## Usage

_By Sarah Badenhorst, Josiah Heng, Isadora Jordan and Robert van Poetern_

