import cv2
from ai_engine.object_detector import ObjectDetector
from processing import functions
import os

# Model targets
OBJECTS = ("face", "hand")
face_detector = ObjectDetector(OBJECTS[0])
hand_detector = ObjectDetector(OBJECTS[1])

# Dimensionality parameters
HOLO_BOX_SIZE = (400, 200)
HEAD_BOX_SIZE = (250, 300)
FACE_BOX_SIZE = (int(0.60 * HEAD_BOX_SIZE[0]), int(0.58 * HEAD_BOX_SIZE[1]))

# Effects
HOLOGRAM_EFFECT_PATH = os.path.join(os.getcwd(), "utility", "holo.mp4")
FACE_BASELINE_PATH = os.path.join(os.getcwd(), "utility", "altra_baseline.png")

# Stream IP source (int for local cam where 0 usually is the webcam included in you laptop or greater,
# you can use url for streaming ex: face_source="urlofavideostream")
hand_source = 0
face_source = 1


def run_hands_pipeline(model, src):

    # Hand wrist coords extraction
    hand_key_points = model.extract_key_points(src)
    if hand_key_points:
        wrist_location_x, wrist_location_y = hand_key_points["wrist"]["x"], hand_key_points["wrist"]["y"]

        hologram_box_coords = functions.location_box_from_coords((wrist_location_x, wrist_location_y),
                                                                 y_lag=180, box_dims=HOLO_BOX_SIZE)
        head_box_coords = functions.location_box_from_coords((wrist_location_x, wrist_location_y),
                                                             y_lag=HEAD_BOX_SIZE[1] + 100, box_dims=HEAD_BOX_SIZE)
        face_box_coords = functions.location_box_from_coords((wrist_location_x, wrist_location_y),
                                                             y_lag=HEAD_BOX_SIZE[1] + 30, box_dims=FACE_BOX_SIZE)

        return hologram_box_coords, head_box_coords, face_box_coords
    else:
        pass


def run_face_pipeline(model, src):
    face_bbox = model.extract_bounding_box(src)
    face_area = functions.crop_img_from_bounding_box(src, face_bbox)
    key_points_face = model.extract_key_points(face_area)
    extraction = functions.crop_face_from_key_points(face_area, key_points_face)

    return extraction


def run_main(face_model, hands_model):
    cap = cv2.VideoCapture(hand_source)  # Background capture live
    face_cap = cv2.VideoCapture(face_source)  # Face capture live
    holo_effect = cv2.VideoCapture(HOLOGRAM_EFFECT_PATH)
    face_baseline = cv2.imread(FACE_BASELINE_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        face_success, face_frame = face_cap.read()
        if not success:
            pass

        try:
            face_extracted = run_face_pipeline(face_model, face_frame)
            holo_effect_area, baseline_head_area, face_area = run_hands_pipeline(hands_model, frame)

            holo_success, holo_frame = holo_effect.read()
            if not holo_success:
                holo_effect.set(cv2.CAP_PROP_POS_FRAMES, 0)

            functions.insert_image(face_baseline, frame, baseline_head_area, 0.2)
            functions.insert_image(cv2.cvtColor(face_extracted, cv2.COLOR_RGB2BGR), frame, face_area, 0.5)
            functions.insert_image_fixed_opacity(holo_frame, frame, holo_effect_area)
        except cv2.error:
            pass
        except TypeError:
            pass

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_main(face_detector, hand_detector)
