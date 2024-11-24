
# EinsteinVision

# Inference :
```bash
    python main.py
```
arguments :
```bash
    python main.py --debug=True
```
will show the logs.

### References:

1. `YOLO3D` : To Detect Bounding boxes and orientations of the vehicles.
2. `Yoloworld from ultralytics` : To Detect Bounding boxes of the types of vehicles, types of traffic lights, person, traffic cone, speed limit sign, road signs, speed breaker/hump,trash can.
Traffic lights, Persons, Stop Sign.
3. `CLRERNet`: To detect Lane Detection. Detects (Solid Line , dashed Line, Double Line).
4. `PyMAF`: Generates human Mesh .obj files for every frames.
5. `EasyOCR`: To detect text on the road signs.
6. `ZoeDepth`: To detect depth of the frames.
7. `RAFT`: To get optical flow b/w each frames.
8. `https://github.com/gsethan17/one-stage-brake-light-status-detection`: To detect car rear lights.