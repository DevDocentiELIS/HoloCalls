import mediapipe as mp
from ai_engine import detections
import cv2
import numpy as np


class ObjectDetector:

    def __init__(self, target_obj: str):
        try:
            self.set_target_object(target_obj)
            self.set_detection_engine()
        except ValueError as e:
            print("[ERROR] An Error occurred while creating the Object Detector model")
            print(f"[ERROR] {e}")

    def instance_from_yaml(self, yaml_src: str):
        pass

    def set_target_object(self, target_obj: str) -> None:
        if target_obj in ('face', 'hand'):
            setattr(self, "__target_object", target_obj)
        else:
            raise ValueError(f'Target object must be either "face" or "hand" not {target_obj}')

    def set_detection_engine(self) -> None:
        model_type = getattr(self, '__target_object')
        if model_type == 'face':
            setattr(self, "__detection_engine", mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.8))
            setattr(self, '__KeyPoints_extractor',
                    mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    )
        elif model_type == 'hand':
            setattr(self, "__detection_engine",  mp.solutions.hands.Hands(max_num_hands=1))
            setattr(self, "__KeyPoints_extractor", getattr(self, '__detection_engine'))
        else:
            raise ValueError(f'Cannot interpreter model type {model_type}, target_obj must be either "face" or "hand"')

    def extract_bounding_box(self, src: np.ndarray) -> dict[str, int]:
        if getattr(self, "__target_object") == "face":
            extraction = detections.Face.from_frame(self, src)
            bounding_b = extraction.bbox
            return bounding_b
        else:
            raise NotImplementedError("Bounding box extraction is only implemented for faces")

    def extract_key_points(self, src: np.ndarray):
        if getattr(self, "__target_object") == "face":
            key_points = detections.Face.from_frame(self, src).key_points
            return key_points
        elif getattr(self, "__target_object") == "hand":
            key_points = detections.Hand.from_frame(self, src).key_points
            return key_points
        else:
            raise NotImplementedError("key points extraction is only implemented for faces and hands")

    def __repr__(self) -> str:
        if hasattr(self, '__target_object'):
            return f'{self.__class__.__name__} [Target Object] {getattr(self, "__target_object")}'
        else:
            return '[ERROR] Object Detector Model not correctly instantiated'
