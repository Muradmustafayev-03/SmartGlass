from face_detection import MPFaceDetection
from engine import Engine

if __name__ == '__main__':
    mpFaceDetector = MPFaceDetection(model_selection=False, confidence=0.5)
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[mpFaceDetector])
    selfieSegmentation.run()
