import math
import random
import sys
import threading
import time
from dataclasses import dataclass, field
import os

import cv2
import mediapipe as mp
import numpy as np
import pygame

WIDTH, HEIGHT = 1280, 720
CAM_W, CAM_H = 320, 240
FPS = 60

MAX_SPEED = 1.5
ACCELERATION = 0.02
FRICTION = 0.01
TURN_SPEED = 0.04
CAM_HEIGHT = 5.0
CAM_DISTANCE = 10.0
FOCAL_LENGTH = 680.0

SKY = (100, 180, 220)
GRASS_DARK = (25, 100, 35)
GRASS_LIGHT = (50, 140, 60)
ASPHALT = (40, 40, 45)
ASPHALT_LIGHT = (80, 80, 90)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (0, 255, 204)
RED = (255, 80, 80)
YELLOW = (255, 235, 59)
DARK_GRAY = (20, 20, 25)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

ASSET_DIR = os.path.join(os.path.dirname(__file__), "3dassest")


def generate_grass_texture(w=256, h=256):
    """Generate a realistic grass texture with procedural noise."""
    tex = np.ones((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    
    for y in range(h):
        for x in range(w):
            noise = rng.rand() * 0.3
            if rng.rand() > 0.7:
                noise += 0.2
            base_g = int(70 + noise * 50)
            tex[y, x] = [int(base_g * 0.5), base_g, int(base_g * 0.6)]
    
    for _ in range(w * h // 100):
        xi, yi = rng.randint(0, w), rng.randint(0, h)
        size = rng.randint(1, 4)
        for dx in range(-size, size):
            for dy in range(-size, size):
                if 0 <= xi + dx < w and 0 <= yi + dy < h:
                    tex[yi + dy, xi + dx] = [40, 100, 35]
    return tex


def generate_road_texture(w=256, h=256):
    """Generate a realistic asphalt road texture."""
    tex = np.ones((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(42)
    
    for y in range(h):
        for x in range(w):
            noise = rng.rand() * 30
            tex[y, x] = [40 + int(noise * 0.3), 40 + int(noise * 0.3), 45 + int(noise * 0.3)]
    
    for _ in range(w * h // 300):
        xi, yi = rng.randint(0, w), rng.randint(0, h)
        size = rng.randint(1, 3)
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                if 0 <= xi + dx < w and 0 <= yi + dy < h:
                    tex[yi + dy, xi + dx] = [50, 50, 55]
    
    for _ in range(3):
        y = rng.randint(h // 4, 3 * h // 4)
        for x in range(w):
            if rng.rand() > 0.5:
                tex[y, x] = [250, 250, 200]
    
    return tex


@dataclass
class BoxObject:
    name: str
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float
    color: tuple[int, int, int]
    rotation: float = 0.0
    kind: str = "box"
    speed: float = 0.0
    extra: dict = field(default_factory=dict)


class WebcamCapture(threading.Thread):
    def __init__(self, src=0, w=CAM_W, h=CAM_H):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.running = True
        self.lock = threading.Lock()
        self.frame = None
        self.error = None

    def run(self):
        if not self.cap.isOpened():
            self.error = "Camera not available"
            return
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.frame = frame
        self.cap.release()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def hand_is_open(hand_landmarks):
    return (hand_landmarks[8].y < hand_landmarks[5].y) and (hand_landmarks[12].y < hand_landmarks[9].y)


def load_obj_model(filepath, max_faces=10000):
    """Load OBJ model efficiently, sampling faces if needed."""
    vertices = []
    faces = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    try:
                        v = (float(parts[1]), float(parts[2]), float(parts[3]))
                        vertices.append(v)
                    except (ValueError, IndexError):
                        continue
                
                elif parts[0] == 'f':
                    try:
                        face_indices = []
                        for part in parts[1:]:
                            vertex_idx = int(part.split('/')[0]) - 1
                            if 0 <= vertex_idx < len(vertices):
                                face_indices.append(vertex_idx)
                        if len(face_indices) >= 3:
                            faces.append(face_indices)
                    except (ValueError, IndexError):
                        continue
        
        # Sample faces if too many
        if len(faces) > max_faces:
            sample_rate = len(faces) // max_faces
            faces = faces[::sample_rate]
        
        print(f"Loaded OBJ: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces
    
    except Exception as e:
        print(f"Error loading OBJ: {e}")
        return [], []


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Gesture Drive City 3D")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Segoe UI", 18)
        self.titlefont = pygame.font.SysFont("Segoe UI", 20, bold=True)

        self.debug_lines: list[str] = []
        self.loader_visible = True
        self.loader_message = "Loading 3D Engine & AI Vision..."
        self.fallback_visible = False
        self.start_time = time.monotonic()
        self.first_frame_received = False
        self.control_mode = "camera"

        self.speed = 0.0
        self.distance = 0.0
        self.steering = 0.0
        self.look_up_offset = 0.0

        self.car_x = 0.0
        self.car_y = 0.0
        self.car_z = 0.0
        self.car_yaw = 0.0

        self.gesture_status = "WAITING FOR HANDS..."
        self.face_status = "Searching..."
        self.gas = False
        self.brake = False
        self.keyboard_state = {"w": False, "a": False, "s": False, "d": False}

        self.webcam = WebcamCapture()
        self.webcam.start()

        self.holistic = mp_holistic.Holistic(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.preview_overlay = None

        self.static_objects: list[BoxObject] = []
        self.traffic_cars: list[BoxObject] = []
        self.pedestrians: list[BoxObject] = []
        self.birds: list[dict] = []
        self.clouds: list[dict] = []
        
        self.grass_texture = None
        self.road_texture = None
        
        # Load 3D car model
        self.car_vertices = []
        self.car_faces = []
        self.car_bounds = None
        self.load_car_model()
        
        self.build_world()

        self.log("Holistic model initialized")

    def log(self, message):
        print(message)
        self.debug_lines.append(message)
        self.debug_lines = self.debug_lines[-7:]

    def load_car_model(self):
        """Load 3D car model from OBJ file."""
        obj_path = os.path.join(ASSET_DIR, "bugatti.obj")
        if not os.path.exists(obj_path):
            obj_path = os.path.join(ASSET_DIR, "car.obj")
        
        if os.path.exists(obj_path):
            self.car_vertices, self.car_faces = load_obj_model(obj_path, max_faces=5000)
            if self.car_vertices:
                # Calculate bounds for scaling
                xs = [v[0] for v in self.car_vertices]
                ys = [v[1] for v in self.car_vertices]
                zs = [v[2] for v in self.car_vertices]
                self.car_bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
                self.log(f"3D Car model loaded: {len(self.car_vertices)} verts, {len(self.car_faces)} faces")
        else:
            self.log("Car model not found, using default box")

    def draw_car_3d(self, cam):
        """Draw the 3D car model at current position."""
        if not self.car_vertices or not self.car_faces:
            # Fallback to simple box
            self.draw_box(BoxObject("player_body", self.car_x, 0.8, self.car_z, 1.8, 0.8, 4, (255, 0, 85)), cam)
            return

        # Scale and transform vertices
        if not self.car_bounds:
            return

        min_x, max_x, min_y, max_y, min_z, max_z = self.car_bounds
        model_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
        scale = 2.0 / model_size
        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        center_z = (max_z + min_z) / 2

        # Precompute rotated & projected vertices
        projected_verts = [None] * len(self.car_vertices)
        cy = math.cos(self.car_yaw)
        sy = math.sin(self.car_yaw)
        for i, v in enumerate(self.car_vertices):
            # Normalize and scale
            x = (v[0] - center_x) * scale
            y = (v[1] - center_y) * scale - 0.25
            z = (v[2] - center_z) * scale

            # Rotate around model Y by car_yaw
            rx = x * cy - z * sy
            rz = x * sy + z * cy

            # Position relative to car
            world_x = self.car_x + rx
            world_y = 0.5 + y
            world_z = self.car_z + rz

            p = self.transform_point((world_x, world_y, world_z), cam)
            projected_verts[i] = p

        # Prepare faces with depth sorting and basic backface culling
        face_items = []
        for face in self.car_faces:
            if len(face) < 3:
                continue
            pts2d = []
            depth_sum = 0.0
            valid = True
            for idx in face:
                if idx < len(projected_verts) and projected_verts[idx]:
                    px, py, pz = projected_verts[idx]
                    pts2d.append((px, py))
                    depth_sum += pz
                else:
                    valid = False
                    break
            if not valid or len(pts2d) < 3:
                continue

            # Backface culling via signed area of projected triangle (first three points)
            ax, ay = pts2d[0]
            bx, by = pts2d[1]
            cx, cy2 = pts2d[2]
            signed_area = (bx - ax) * (cy2 - ay) - (by - ay) * (cx - ax)
            if signed_area >= 0:
                # face is back-facing (winding dependent) - skip
                continue

            avg_depth = depth_sum / len(face)
            face_items.append((avg_depth, pts2d, face))

        # Painter's algorithm: draw farthest faces first
        face_items.sort(key=lambda x: x[0], reverse=True)

        for avg_depth, pts2d, face in face_items:
            # Convert to int points
            pts = [(int(x), int(y)) for (x, y) in pts2d]
            # Simple lighting: approximate by depth (darker when farther)
            light = clamp(1.2 - (avg_depth / 40.0), 0.4, 1.0)
            base_color = (200, 60, 100)
            color = (int(base_color[0] * light), int(base_color[1] * light), int(base_color[2] * light))
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, DARK_GRAY, pts, 1)

    def build_world(self):
        """Build an enhanced city environment with realistic details."""
        rng = random.Random(7)

        self.grass_texture = generate_grass_texture()
        self.road_texture = generate_road_texture()

        self.static_objects.append(BoxObject("hospital_main", -80, 15, -100, 50, 30, 10, (170, 170, 170)))
        self.static_objects.append(BoxObject("hospital_left", -115, 15, -95, 20, 30, 15, (136, 204, 255)))
        self.static_objects.append(BoxObject("hospital_right", -45, 15, -95, 20, 30, 15, (136, 204, 255)))
        self.static_objects.append(BoxObject("hospital_logo", -80, 20, -89.8, 40, 20, 0.2, WHITE, kind="banner"))
        self.static_objects.append(BoxObject("welcome_left", -14, 6, 25, 1, 12, 1, (51, 51, 51), kind="pillar"))
        self.static_objects.append(BoxObject("welcome_right", 14, 6, 25, 1, 12, 1, (51, 51, 51), kind="pillar"))
        self.static_objects.append(BoxObject("welcome_board", 0, 9, 25, 30, 7.5, 0.2, (0, 34, 68), kind="banner"))
        self.static_objects.append(BoxObject("statue_pedestal", 40, 3, -50, 4, 6, 4, (85, 85, 85)))
        self.static_objects.append(BoxObject("statue_torso", 40, 11, -50, 2.5, 4, 1.5, (205, 127, 50)))
        self.static_objects.append(BoxObject("statue_head", 40, 14, -50, 1.5, 1.5, 1.5, (205, 127, 50), kind="sphere"))
        self.static_objects.append(BoxObject("statue_plaque", 40, 4, -48.9, 3.8, 2, 0.2, (51, 51, 51), kind="banner"))

        for i in range(-20, 25, 5):
            self.static_objects.append(BoxObject(f"bush_{i}", i, 1, -95, 1.5, 1.5, 1.5, (34, 139, 34), kind="sphere"))

        for i in range(-12, 13):
            for offset in [0, 60]:
                z_pos = i * 60 + offset
                self.static_objects.append(BoxObject(f"road_v_{i}_{offset}", i * 60, 0, z_pos, 10, 0.1, 60, ASPHALT, kind="road"))

            if i == 0 or i == -1 or i == 1:
                continue

            for j in range(-12, 13):
                zone_x = i * 60
                zone_z = j * 60

                if i == -1 and abs(j * 60 + 100) < 40:
                    continue
                if i == 1 and abs(j * 60 + 50) < 50:
                    continue

                if rng.random() > 0.25:
                    h = rng.uniform(8, 25)
                    w = rng.uniform(8, 16)
                    d = rng.uniform(8, 16)
                    color = rng.choice([(100, 100, 120), (120, 110, 100), (90, 95, 110), (140, 130, 120)])
                    self.static_objects.append(BoxObject(f"building_{i}_{j}", zone_x + rng.uniform(-15, 15), h / 2, zone_z + rng.uniform(-15, 15), w, h, d, color))

                for _ in range(rng.randint(0, 2)):
                    tree_x = zone_x + rng.uniform(-25, 25)
                    tree_z = zone_z + rng.uniform(-25, 25)
                    self.static_objects.append(BoxObject(f"tree_{i}_{j}_{_}", tree_x, 0, tree_z, 3, 7, 3, (139, 69, 19), kind="tree"))

                for _ in range(rng.randint(0, 3)):
                    lamp_x = zone_x + rng.choice([-20, 20])
                    lamp_z = zone_z + rng.uniform(-25, 25)
                    self.static_objects.append(BoxObject(f"lamp_{i}_{j}_{_}", lamp_x, 0, lamp_z, 0.5, 8, 0.5, (40, 40, 40), kind="lamp"))

        car_colors = [(255, 0, 0), (0, 100, 255), (255, 200, 0), (100, 255, 100), (255, 100, 255), (200, 200, 200)]
        for _ in range(40):
            lane_x = rng.randint(-8, 8) * 60
            self.traffic_cars.append(BoxObject("traffic", lane_x, 0.5, rng.uniform(-250, 250), 2, 0.9, 4.5, rng.choice(car_colors), speed=rng.uniform(0.15, 0.5)))

        for _ in range(50):
            self.pedestrians.append(BoxObject("pedestrian", rng.uniform(-300, 300), 0.9, rng.uniform(-300, 300), 0.5, 1.7, 0.5, rng.choice([(255, 170, 0), (200, 100, 50), (150, 150, 200), (220, 100, 100)]), kind="pedestrian", extra={"dx": rng.uniform(-0.08, 0.08), "dz": rng.uniform(-0.08, 0.08)}))

        for _ in range(25):
            self.birds.append({"x": rng.uniform(-150, 150), "y": rng.uniform(25, 70), "z": rng.uniform(-150, 150), "speed": rng.uniform(0.15, 0.35), "angle": rng.uniform(0, math.tau)})

        for _ in range(35):
            self.clouds.append({"x": rng.uniform(-300, 300), "y": rng.uniform(50, 90), "z": rng.uniform(-300, 300), "scale": rng.uniform(1.0, 2.5)})


    def process_frame(self, frame):
        frame = cv2.resize(frame, (CAM_W, CAM_H))
        mirrored = cv2.flip(frame, 1)
        results = self.holistic.process(mirrored)

        status = "NEED 2 HANDS"
        face_status = "NOT DETECTED"
        gesture_throttle = 0
        gesture_turn = 0
        look_up_offset = 0.0

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(mirrored, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(mirrored, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        left_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
        right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

        if left_hand and right_hand:
            w1 = left_hand[0]
            w2 = right_hand[0]
            angle = math.degrees(math.atan2(w2.y - w1.y, w2.x - w1.x))
            if angle > 15:
                gesture_turn = -1
                status = "TURNING RIGHT"
            elif angle < -15:
                gesture_turn = 1
                status = "TURNING LEFT"
            else:
                status = "STRAIGHT"

            if hand_is_open(left_hand) and hand_is_open(right_hand):
                gesture_throttle = 1
                status += " - GOING"
            elif not hand_is_open(left_hand) and not hand_is_open(right_hand):
                gesture_throttle = -1
                status += " - BRAKING"
        else:
            status = "NEED 2 HANDS"

        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            nose = landmarks[4]
            top_head = landmarks[10]
            chin = landmarks[152]
            face_height = chin.y - top_head.y
            nose_rel_y = (nose.y - top_head.y) / face_height if face_height else 1.0

            if nose_rel_y < 0.4:
                look_up_offset = min((0.4 - nose_rel_y) * 5.0, 2.0)
                face_status = "LOOKING UP"
            else:
                face_status = "DETECTED"

        self.preview_overlay = mirrored
        self.gesture_status = status
        self.face_status = face_status
        self.gas = gesture_throttle == 1
        self.brake = gesture_throttle == -1
        self.steering = gesture_turn
        self.look_up_offset = look_up_offset

        return results

    def handle_keyboard(self):
        keys = pygame.key.get_pressed()
        self.keyboard_state["w"] = keys[pygame.K_w]
        self.keyboard_state["a"] = keys[pygame.K_a]
        self.keyboard_state["s"] = keys[pygame.K_s]
        self.keyboard_state["d"] = keys[pygame.K_d]
        self.gas = self.keyboard_state["w"]
        self.brake = self.keyboard_state["s"]
        self.steering = -1 if self.keyboard_state["a"] else (1 if self.keyboard_state["d"] else 0)
        self.gesture_status = "KEYBOARD MODE"

    def update_physics(self):
        target_speed = 0.0
        if self.gas:
            target_speed = MAX_SPEED
        elif self.brake:
            target_speed = -MAX_SPEED / 2

        if target_speed > self.speed:
            self.speed = min(MAX_SPEED, self.speed + ACCELERATION)
        elif target_speed < self.speed:
            self.speed = max(-MAX_SPEED / 2, self.speed - ACCELERATION)

        if not self.gas and not self.brake:
            self.speed *= 0.95

        if abs(self.speed) < 0.01:
            self.speed = 0.0

        self.car_yaw += self.steering * TURN_SPEED * (self.speed / MAX_SPEED if MAX_SPEED else 0)
        self.distance += abs(self.speed)

        forward_x = math.sin(self.car_yaw)
        forward_z = math.cos(self.car_yaw)
        self.car_x += forward_x * self.speed
        self.car_z += forward_z * self.speed

        for car in self.traffic_cars:
            car.z += car.speed
            if car.z > 200:
                car.z = -200
            if car.z < -200:
                car.z = 200

        for ped in self.pedestrians:
            ped.x += ped.extra["dx"]
            ped.z += ped.extra["dz"]
            if random.random() < 0.02:
                ped.extra["dx"] = random.uniform(-0.1, 0.1)
                ped.extra["dz"] = random.uniform(-0.1, 0.1)

        for bird in self.birds:
            bird["x"] += math.cos(bird["angle"]) * bird["speed"]
            bird["z"] += math.sin(bird["angle"]) * bird["speed"]
            if abs(bird["x"]) > 200:
                bird["x"] *= -1
            if abs(bird["z"]) > 200:
                bird["z"] *= -1

    def camera_state(self):
        forward_x = math.sin(self.car_yaw)
        forward_z = math.cos(self.car_yaw)
        return self.car_x - forward_x * CAM_DISTANCE, CAM_HEIGHT, self.car_z - forward_z * CAM_DISTANCE, self.car_yaw

    def transform_point(self, point, cam):
        cam_x, cam_y, cam_z, cam_yaw = cam
        dx = point[0] - cam_x
        dy = point[1] - cam_y
        dz = point[2] - cam_z

        cos_y = math.cos(-cam_yaw)
        sin_y = math.sin(-cam_yaw)
        x1 = dx * cos_y - dz * sin_y
        z1 = dx * sin_y + dz * cos_y

        if z1 <= 0.35:
            return None

        horizon = HEIGHT * 0.42 + self.look_up_offset * 36
        sx = WIDTH / 2 + x1 * FOCAL_LENGTH / z1
        sy = horizon - dy * FOCAL_LENGTH / z1
        return sx, sy, z1

    def draw_box(self, box, cam, fill_color=None, line_color=None):
        fill_color = fill_color or box.color
        # Darker outline for better definition
        line_color = line_color or tuple(max(0, c - 60) for c in fill_color)

        corners = []
        for dx in (-box.w / 2, box.w / 2):
            for dy in (0, box.h):
                for dz in (-box.d / 2, box.d / 2):
                    corners.append(self.transform_point((box.x + dx, box.y + dy, box.z + dz), cam))

        if not any(corners):
            return

        idx = lambda a, b, c: a * 4 + b * 2 + c
        # Front face, back face, top face with better depth shading
        faces = [
            ([idx(0, 0, 0), idx(1, 0, 0), idx(1, 1, 0), idx(0, 1, 0)], 1.0),
            ([idx(0, 0, 1), idx(1, 0, 1), idx(1, 1, 1), idx(0, 1, 1)], 0.65),
            ([idx(0, 1, 0), idx(1, 1, 0), idx(1, 1, 1), idx(0, 1, 1)], 0.80),
        ]

        for indices, shade in faces:
            pts = []
            for i in indices:
                if corners[i] is None:
                    pts = []
                    break
                pts.append((int(corners[i][0]), int(corners[i][1])))
            if not pts or len(pts) < 3:
                continue
            # Apply depth shading for better 3D appearance
            shaded = tuple(clamp(int(c * shade), 0, 255) for c in fill_color)
            pygame.draw.polygon(self.screen, shaded, pts)
            pygame.draw.polygon(self.screen, line_color, pts, 2)

    def draw_sphere(self, box, cam):
        projected = self.transform_point((box.x, box.y + box.h / 2, box.z), cam)
        if not projected:
            return
        sx, sy, depth = projected
        radius = max(3, int(260 / depth))
        # Draw with outline for better definition
        pygame.draw.circle(self.screen, box.color, (int(sx), int(sy)), radius)
        pygame.draw.circle(self.screen, tuple(max(0, c - 60) for c in box.color), (int(sx), int(sy)), radius, 1)

    def draw_tree(self, box, cam):
        self.draw_box(BoxObject(box.name + "_trunk", box.x, box.y + 1.5, box.z, 0.8, 3, 0.8, (101, 50, 20)), cam)
        self.draw_sphere(BoxObject(box.name + "_leaves", box.x, box.y + 5.5, box.z, 5, 6, 5, (34, 120, 34), kind="sphere"), cam)

    def draw_lamp(self, box, cam):
        """Draw a street lamp pole with light."""
        pole = BoxObject(box.name + "_pole", box.x, box.y + 4, box.z, 0.3, 8, 0.3, (30, 30, 30))
        self.draw_box(pole, cam)
        light_head = BoxObject(box.name + "_head", box.x, box.y + 8.2, box.z, 0.8, 0.8, 0.8, (255, 250, 200), kind="sphere")
        self.draw_sphere(light_head, cam)


    def draw_banner(self, box, cam, lines, bg=(0, 34, 68), border=(255, 215, 0), text_color=WHITE):
        left_top = self.transform_point((box.x - box.w / 2, box.y + box.h / 2, box.z), cam)
        right_bottom = self.transform_point((box.x + box.w / 2, box.y - box.h / 2, box.z), cam)
        if not left_top or not right_bottom:
            return
        x1, y1, _ = left_top
        x2, y2, _ = right_bottom
        rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        # Draw banner background with padding
        pygame.draw.rect(self.screen, bg, rect)
        pygame.draw.rect(self.screen, border, rect, 4)
        # Center text with proper spacing
        line_height = 24
        total_height = len(lines) * line_height
        start_y = rect.centery - total_height // 2 + line_height // 2
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, text_color)
            y = start_y + i * line_height
            self.screen.blit(surf, surf.get_rect(center=(rect.centerx, y)))

    def draw_hospital(self, cam):
        self.draw_box(BoxObject("hospital_main", -80, 15, -100, 50, 30, 10, (170, 170, 170)), cam)
        self.draw_box(BoxObject("hospital_left", -115, 15, -95, 20, 30, 15, (136, 204, 255)), cam)
        self.draw_box(BoxObject("hospital_right", -45, 15, -95, 20, 30, 15, (136, 204, 255)), cam)
        self.draw_banner(BoxObject("hospital_logo", -80, 20, -89.8, 40, 20, 0.2, WHITE, kind="banner"), cam, ["University Hospital KDU"])
        for i in range(-20, 25, 5):
            self.draw_sphere(BoxObject("bush", i, 1, -95, 1.5, 1.5, 1.5, (34, 139, 34), kind="sphere"), cam)

    def draw_statue(self, cam):
        self.draw_box(BoxObject("pedestal", 40, 3, -50, 4, 6, 4, (85, 85, 85)), cam)
        for part in [
            BoxObject("leg_l", 39.2, 7, -50, 0.8, 4, 0.8, (205, 127, 50)),
            BoxObject("leg_r", 40.8, 7, -50, 0.8, 4, 0.8, (205, 127, 50)),
            BoxObject("torso", 40, 11, -50, 2.5, 4, 1.5, (205, 127, 50)),
            BoxObject("head", 40, 14, -50, 1.5, 1.5, 1.5, (205, 127, 50), kind="sphere"),
            BoxObject("arm_l", 38.2, 11, -50, 0.7, 3.5, 0.7, (205, 127, 50)),
            BoxObject("arm_r", 41.8, 11, -50, 0.7, 3.5, 0.7, (205, 127, 50)),
        ]:
            if part.kind == "sphere":
                self.draw_sphere(part, cam)
            else:
                self.draw_box(part, cam)
        self.draw_banner(BoxObject("statue_banner", 40, 4, -48.9, 3.8, 2, 0.2, (51, 51, 51), kind="banner"), cam, ["MRA HASEN AL BANNA", "TEAM BLACK CAP XTREME"], bg=(51, 51, 51), border=(218, 165, 32), text_color=(218, 165, 32))

    def draw_road_network(self, cam):
        """Simplified field + single centered road for clarity and performance.

        This replaces the procedural grass tiles with a solid green field and
        draws a single straight road centered on the X axis using a projected
        quadrilateral. The center dashed line is drawn by projecting points
        along the Z axis.
        """
        # Sky background
        pygame.draw.rect(self.screen, SKY, (0, 0, WIDTH, HEIGHT))

        # Solid green field area under and beyond the horizon
        field_top = int(HEIGHT * 0.42 + self.look_up_offset * 20)
        pygame.draw.rect(self.screen, GRASS_LIGHT, (0, field_top, WIDTH, HEIGHT - field_top))

        # Road parameters in world coordinates
        road_half_width = 6.0
        road_z_near = -400
        road_z_far = 400

        # Project four corners of the road rectangle
        corners = []
        for wx, wz in [(-road_half_width, road_z_near), (road_half_width, road_z_near), (road_half_width, road_z_far), (-road_half_width, road_z_far)]:
            p = self.transform_point((wx, 0.01, wz), cam)
            if p:
                corners.append((int(p[0]), int(p[1])))
            else:
                corners = []
                break

        if len(corners) == 4:
            pygame.draw.polygon(self.screen, ASPHALT, corners)
            pygame.draw.lines(self.screen, DARK_GRAY, True, corners, 4)

            # Draw center dashed line: project short segments along Z
            z = road_z_near
            while z < road_z_far:
                p1 = self.transform_point((0, 0.02, z), cam)
                p2 = self.transform_point((0, 0.02, z + 10), cam)
                if p1 and p2:
                    pygame.draw.line(self.screen, YELLOW, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 6)
                z += 30


    def draw_car(self, cam):
        """Draw the player car using 3D model if available."""
        if self.car_vertices and self.car_faces:
            self.draw_car_3d(cam)
        else:
            # Fallback to simple box representation
            self.draw_box(BoxObject("player_body", self.car_x, 0.8, self.car_z, 1.8, 0.8, 4, (255, 0, 85)), cam)
            self.draw_box(BoxObject("player_roof", self.car_x, 1.5, self.car_z - 0.2, 1.6, 0.6, 2, (34, 34, 34)), cam)
            self.draw_box(BoxObject("hl_left", self.car_x - 0.6, 0.8, self.car_z + 2.0, 0.4, 0.2, 0.1, (255, 255, 170)), cam)
            self.draw_box(BoxObject("hl_right", self.car_x + 0.6, 0.8, self.car_z + 2.0, 0.4, 0.2, 0.1, (255, 255, 170)), cam)

    def draw_traffic(self, cam):
        for car in self.traffic_cars:
            self.draw_box(car, cam)

    def draw_pedestrians(self, cam):
        for ped in self.pedestrians:
            self.draw_box(BoxObject(ped.name, ped.x, ped.y, ped.z, 0.6, 1.5, 0.6, ped.color, kind="pedestrian"), cam)

    def draw_birds(self, cam):
        for bird in self.birds:
            p = self.transform_point((bird["x"], bird["y"], bird["z"]), cam)
            if not p:
                continue
            sx, sy, depth = p
            wing = 8 / max(1.0, depth / 20)
            flap = math.sin(time.time() * 10) * wing * 0.5
            pts = [(sx - wing, sy + flap), (sx, sy - wing * 0.15), (sx + wing, sy + flap)]
            pygame.draw.lines(self.screen, BLACK, False, pts, 2)

    def draw_clouds(self, cam):
        for cloud in self.clouds:
            p = self.transform_point((cloud["x"], cloud["y"], cloud["z"]), cam)
            if not p:
                continue
            sx, sy, depth = p
            radius = max(4, int(18 * cloud["scale"] / max(0.5, depth / 40)))
            for off in [(-radius, 0), (0, -radius // 2), (radius, 0), (radius // 2, radius // 2), (-radius // 2, radius // 2)]:
                pygame.draw.circle(self.screen, WHITE, (int(sx + off[0]), int(sy + off[1])), radius)

    def draw_preview(self, results):
        if self.preview_overlay is None:
            return
        preview = self.preview_overlay.copy()
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(preview, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(preview, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
        preview = cv2.resize(preview, (CAM_W, CAM_H))
        surf = pygame.surfarray.make_surface(np.transpose(preview, (1, 0, 2)))
        self.screen.blit(surf, (WIDTH - CAM_W - 20, HEIGHT - CAM_H - 20))
        pygame.draw.rect(self.screen, (30, 30, 30), (WIDTH - CAM_W - 24, HEIGHT - CAM_H - 24, CAM_W + 8, CAM_H + 8), 3)

    def draw_hud(self):
        # Main HUD panel
        hud_rect = pygame.Rect(15, 15, 280, 175)
        pygame.draw.rect(self.screen, (10, 10, 15), hud_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 200, 255), hud_rect, 2, border_radius=8)

        # Title
        title_surf = self.titlefont.render("Gesture Racer", True, CYAN)
        self.screen.blit(title_surf, (30, 25))
        pygame.draw.line(self.screen, (100, 200, 255), (30, 48), (280, 48), 1)

        # Stats with consistent spacing
        self.screen.blit(self.font.render(f"Speed: {math.floor(abs(self.speed) * 100)} km/h", True, YELLOW), (30, 58))
        self.screen.blit(self.font.render(f"Distance: {math.floor(self.distance)} m", True, YELLOW), (30, 82))
        face_color = CYAN if self.face_status != "NOT DETECTED" else RED
        self.screen.blit(self.font.render(f"Face: {self.face_status}", True, face_color), (30, 106))

        # Gesture status box - properly centered
        box_color = CYAN if self.gas else (RED if self.brake else (200, 200, 200))
        gesture_box = pygame.Rect(25, 135, 250, 40)
        pygame.draw.rect(self.screen, (30, 30, 40), gesture_box, border_radius=5)
        pygame.draw.rect(self.screen, box_color, gesture_box, 1, border_radius=5)
        gesture_surf = self.font.render(self.gesture_status, True, box_color)
        gesture_rect = gesture_surf.get_rect(center=gesture_box.center)
        self.screen.blit(gesture_surf, gesture_rect)

        # Instructions at bottom with background
        instr = "Hold invisible steering wheel • Rotate to Turn • Open Hands = GAS • Fists = BRAKE"
        if self.control_mode == "keyboard":
            instr = "WASD to Drive • Arrow Keys to Turn"
        instr_surf = self.font.render(instr, True, WHITE)
        instr_rect = instr_surf.get_rect(center=(WIDTH // 2, HEIGHT - 20))
        # Add dark background to instructions
        bg_rect = instr_rect.inflate(20, 10)
        pygame.draw.rect(self.screen, (10, 10, 15), bg_rect, border_radius=4)
        pygame.draw.rect(self.screen, (100, 100, 110), bg_rect, 1, border_radius=4)
        self.screen.blit(instr_surf, instr_rect)

    def draw_loader(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 245))
        self.screen.blit(overlay, (0, 0))

        spinner_angle = (time.monotonic() - self.start_time) * 360
        center = (WIDTH // 2, HEIGHT // 2 - 40)
        pygame.draw.circle(self.screen, (50, 50, 50), center, 28, 5)
        end = (center[0] + math.cos(math.radians(spinner_angle)) * 28, center[1] + math.sin(math.radians(spinner_angle)) * 28)
        pygame.draw.line(self.screen, CYAN, center, end, 5)

        loading = self.font.render(self.loader_message, True, WHITE)
        self.screen.blit(loading, loading.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 5)))

        y = HEIGHT // 2 + 40
        for line in self.debug_lines[-5:]:
            text = pygame.font.SysFont("Segoe UI", 11).render(line, True, (170, 170, 170))
            self.screen.blit(text, text.get_rect(center=(WIDTH // 2, y)))
            y += 15

        if self.fallback_visible:
            button = pygame.Rect(0, 0, 320, 38)
            button.center = (WIDTH // 2, HEIGHT // 2 + 120)
            pygame.draw.rect(self.screen, (200, 40, 40), button, border_radius=6)
            pygame.draw.rect(self.screen, WHITE, button, 2, border_radius=6)
            label = self.font.render("Camera failed? Click for Keyboard Mode", True, WHITE)
            self.screen.blit(label, label.get_rect(center=button.center))

    def update_loader(self):
        if self.loader_visible and (time.monotonic() - self.start_time) > 8:
            self.loader_message = "Taking a while... Check Camera Permissions."
            self.fallback_visible = True

    def enable_keyboard_mode(self):
        self.control_mode = "keyboard"
        self.loader_visible = False
        self.gesture_status = "KEYBOARD MODE"
        self.log("Keyboard mode enabled")

    def run(self):
        running = True
        results = None
        camera_failed = False

        while running:
            self.clock.tick(FPS)
            self.update_loader()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_k:
                        self.enable_keyboard_mode()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.fallback_visible:
                    button = pygame.Rect(0, 0, 320, 38)
                    button.center = (WIDTH // 2, HEIGHT // 2 + 120)
                    if button.collidepoint(event.pos):
                        self.enable_keyboard_mode()

            frame = self.webcam.read()
            if frame is not None and self.control_mode == "camera":
                if not self.first_frame_received:
                    self.log("Holistic model loaded! Game started.")
                    self.first_frame_received = True
                    self.loader_visible = False
                results = self.process_frame(frame)
            else:
                if self.webcam.error and not camera_failed:
                    self.loader_message = "Camera Access Denied or Error."
                    self.fallback_visible = True
                    camera_failed = True
                    self.log("Camera error: " + self.webcam.error)
                if self.control_mode == "keyboard":
                    self.handle_keyboard()
                elif any(pygame.key.get_pressed()[key] for key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d)):
                    self.enable_keyboard_mode()
                    self.handle_keyboard()

            self.update_physics()
            cam = self.camera_state()

            self.draw_road_network(cam)
            self.draw_clouds(cam)

            for obj in self.static_objects:
                if obj.name == "kdu_hospital":
                    self.draw_hospital(cam)
                elif obj.name == "statue_marker":
                    self.draw_statue(cam)
                elif obj.kind == "pillar":
                    self.draw_box(obj, cam, fill_color=(51, 51, 51))
                elif obj.kind == "banner" and obj.name == "welcome_board":
                    self.draw_banner(obj, cam, ["WELCOME TO", "GENERAL SIR JOHN KOTELAWALA", "DEFENCE UNIVERSITY OF SRI LANKA"])
                elif obj.kind == "banner" and obj.name == "hospital_logo":
                    self.draw_banner(obj, cam, ["University Hospital KDU"])
                elif obj.kind == "banner" and obj.name == "statue_plaque":
                    self.draw_banner(obj, cam, ["MRA HASEN AL BANNA", "TEAM BLACK CAP XTREME"], bg=(51, 51, 51), border=(218, 165, 32), text_color=(218, 165, 32))
                elif obj.kind == "tree" and obj.name.startswith("tree_"):
                    self.draw_tree(obj, cam)
                elif obj.kind == "lamp" and obj.name.startswith("lamp_"):
                    self.draw_lamp(obj, cam)
                elif obj.kind == "sphere":
                    self.draw_sphere(obj, cam)

            self.draw_traffic(cam)
            self.draw_pedestrians(cam)
            self.draw_birds(cam)
            self.draw_car(cam)
            self.draw_hud()

            if results is not None:
                self.draw_preview(results)

            if self.loader_visible:
                self.draw_loader()

            pygame.display.flip()

        self.shutdown()

    def shutdown(self):
        self.webcam.stop()
        self.holistic.close()
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    try:
        game.run()
    except Exception as exc:
        print("Error:", exc)
        game.shutdown()
        sys.exit(1)
