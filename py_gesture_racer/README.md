Gesture Racer (Python) - Enhanced Edition

A full Python reimplementation of the browser `Gesture Racer` demo with stunning procedurally-generated environments, realistic textures, and dynamic city simulation.

## Features

- **Gesture Controls**: Two-hand MediaPipe Holistic tracking for steering and throttle
- **Face Tracking**: Look-up detection for camera elevation
- **Realistic Environment**:
  - Procedurally-generated grass textures with natural variation
  - Detailed asphalt roads with weathering and lane markings
  - 40 dynamic traffic vehicles with varied colors
  - 50 pedestrians with realistic AI navigation
  - Street lampposts with lighting effects
  - Diverse architecture with building variety
  - Natural trees with realistic canopy rendering
  - Bird flocks and dynamic clouds
- **Dual Control**: Gesture recognition or keyboard fallback (WASD)
- **3D Perspective**: Depth-based rendering with realistic proportions and shading
- **Live HUD**: Speed, distance traveled, and status display
- **Webcam Preview**: Real-time hand detection overlay with skeleton tracking

## Requirements

- Python 3.8+
- See `requirements.txt`

## Install

```bash
python -m pip install -r py_gesture_racer/requirements.txt
```

## Run

```bash
python py_gesture_racer/main.py
```

## Controls

**Gesture Mode:**

- Position both hands as if holding an invisible steering wheel
- Rotate hands left/right to steer (based on hand angle)
- Open both hands (palms extended) = **GAS**
- Close both hands (fists clenched) = **BRAKE**
- Tilt head up to look upward in the scene

**Keyboard Mode (fallback):**

- `W` = Accelerate
- `S` = Brake
- `A` = Steer Left
- `D` = Steer Right
- `K` = Force Keyboard Mode
- `ESC` = Exit Game

## Environment Highlights

- **Hospital Complex**: Multi-building medical facility with green signage
- **Welcome Arch**: Iconic entrance to General Sir John Kotelawala Defence University
- **Historical Statue**: Monument and gathering point in the city
- **Dynamic Traffic**: Varied colored vehicles with realistic AI movement
- **Vibrant Pedestrians**: Walking NPCs throughout the city with individual paths
- **Realistic Sky**: Atmospheric perspective with clouds and depth-based rendering
- **Street Furniture**: Lampposts, building variety, and urban details

## Technical Details

- **Rendering**: Custom 3D projection pipeline for perspective-correct depth
- **Textures**: Procedurally generated (no external assets needed)
- **Performance**: Optimized for real-time gesture processing
- **Vision**: MediaPipe Holistic for hand, face, and body tracking
- **Physics**: Simplified but realistic vehicle dynamics

## Notes

- This Python implementation achieves visual parity with the web version
- All environments are procedurally created for lightweight deployment
- Gesture recognition works best with good lighting and clear hand visibility
- Camera must have permission to access for gesture controls to work
