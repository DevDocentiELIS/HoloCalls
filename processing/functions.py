from ai_engine.object_detector import ObjectDetector
import cv2
import numpy as np


def crop_img_from_bounding_box(img, bounding_box: dict[str, int]):
    required_keys = {"x_min", "y_min", "x_max", "y_max"}
    if not bounding_box.keys() == required_keys:
        raise ValueError("Bounding box must have keys {}".format(required_keys))
    cropped_image = img[bounding_box["y_min"]:bounding_box["y_max"], bounding_box["x_min"]:bounding_box["x_max"]]
    return cropped_image


def crop_face_from_key_points(img, face_key_points: dict[str, int]):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(face_key_points)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    return cv2.bitwise_and(img, img, mask=mask)


def location_box_from_coords(starting_point: tuple[int, int], y_lag: int = 0,
                             x_lag: int = 0, box_dims: tuple[int, int] = (0, 0)):

    box_coords = {}
    box_actual_center = (starting_point[0] - x_lag, starting_point[1] - y_lag)

    coords = {
        "x1": box_actual_center[0] + int(0.5 * box_dims[0]),
        "y1": box_actual_center[1],
        "x2": box_actual_center[0] - int(0.5 * box_dims[0]),
        "y2": box_actual_center[1] + box_dims[1],
    }

    box_coords["x_min"], box_coords["y_min"] = min(coords["x1"], coords["x2"]), min(coords["y1"], coords["y2"])
    box_coords["x_max"], box_coords["y_max"] = max(coords["x1"], coords["x2"]), max(coords["y1"], coords["y2"])

    return box_coords


def insert_image(img: np.ndarray, background_img: np.ndarray, location: dict, alpha: float):
    beta = 1 - alpha
    gamma = 0.4
    try:
        to_insert = cv2.resize(img, (location["x_max"] - location["x_min"], location["y_max"] - location["y_min"]))
        background_img[location["y_min"]:location["y_max"], location["x_min"]:location["x_max"]] = cv2.addWeighted(
            background_img[location["y_min"]:location["y_max"], location["x_min"]:location["x_max"]], alpha,
            np.where(to_insert != (0, 0, 0),
                     to_insert,
                     background_img[location["y_min"]:location["y_max"], location["x_min"]:location["x_max"]]
                     ), beta, gamma)
    except cv2.error:
        pass
    except ValueError:
        pass


def insert_image_fixed_opacity(img: np.ndarray, background_img: np.ndarray, location: dict):
    try:
        to_insert = cv2.resize(img, (location["x_max"] - location["x_min"], location["y_max"] - location["y_min"]))
        background_img[location["y_min"]:location["y_max"], location["x_min"]:location["x_max"]] = cv2.add(
            background_img[location["y_min"]:location["y_max"], location["x_min"]:location["x_max"]], to_insert)
    except cv2.error:
        print("errore cv2 nella funzione insert image fixed opacity")
        pass
