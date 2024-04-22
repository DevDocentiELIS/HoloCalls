import cv2
import numpy as np
from ai_engine import object_detector


class Detection:
    def __init__(self):
        self.bbox = {
            "x_min": 0,
            "y_min": 0,
            "x_max": 0,
            "y_max": 0,
        }
        self.key_points = []


class Face(Detection):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_frame(model, frame):
        instance = Face()
        instance.set_bbox(model, frame)
        instance.set_key_points(model, frame)

        return instance

    def set_bbox(self, model: object_detector.ObjectDetector, src):
        to_process = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        face_extraction = getattr(model, "__detection_engine").process(to_process)
        if face_extraction.detections:
            face = face_extraction.detections[0]
            bbox_coords = face.location_data.relative_bounding_box
            ih, iw, _ = src.shape
            bbox = int(bbox_coords.xmin * iw), int(bbox_coords.ymin * ih), \
                int(bbox_coords.width * iw), int(bbox_coords.height * ih)
            self.bbox["x_min"] = bbox[0]
            self.bbox["y_min"] = bbox[1]
            self.bbox["x_max"] = bbox[0] + bbox[2]
            self.bbox["y_max"] = bbox[1] + bbox[3]

    def set_key_points(self, model: object_detector.ObjectDetector, src):
        extractor = getattr(model, "__KeyPoints_extractor")
        to_process = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        face_extraction = extractor.process(to_process)
        face_points = None
        if face_extraction.multi_face_landmarks:
            face_points = np.array([(int(point.x * src.shape[1]), int(point.y * src.shape[0]))
                                    for point in face_extraction.multi_face_landmarks[0].landmark])
        self.key_points = face_points


class Hand(Detection):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_frame(model, frame):
        instance = Hand()
        instance.set_key_points(model, frame)

        return instance

    def set_key_points(self, model: object_detector.ObjectDetector, src):
        extractor = getattr(model, "__KeyPoints_extractor")
        to_process = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        hand_extraction = extractor.process(to_process)
        hand_key_points = {}

        def mark_keypoint_coordinate(hand_zone_index):
            spacial_coordinates = hand_extraction.multi_hand_landmarks[0].landmark[hand_zone_index]
            key_point_x, key_point_y = map(int, (src.shape[1] * spacial_coordinates.x,
                                                 src.shape[0] * spacial_coordinates.y))
            return key_point_x, key_point_y

        if hand_extraction.multi_hand_landmarks:
            (hand_key_points["thumb"],
             hand_key_points["index"],
             hand_key_points["middle"],
             hand_key_points["ring"],
             hand_key_points["pinky"],
             hand_key_points["wrist"]) = {}, {}, {}, {}, {}, {}

            hand_key_points["wrist"]["x"], hand_key_points["wrist"]["y"] = mark_keypoint_coordinate(0)

            fingers = ("thumb", "index", "middle", "ring", "pinky")
            for finger in fingers:
                for index, finger_zone in enumerate(("mcp", "pip", "dip", "tip")):
                    hand_key_points[finger][finger_zone] = {}
                    digit_part_index = (index + 1) * (fingers.index(finger) + 1)
                    (hand_key_points[finger][finger_zone]["x"],
                        hand_key_points[finger][finger_zone]["y"]) = mark_keypoint_coordinate(digit_part_index)

            self.key_points = hand_key_points
