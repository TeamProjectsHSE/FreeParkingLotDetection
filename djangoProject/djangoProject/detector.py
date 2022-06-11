import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from djangoProject import settings

path_to_weights = settings.BASE_DIR.__str__() + R'\runs\train\yolov5x_results\weights\best.pt'


class Detector:

    def __init__(self, path_to_weights):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights, force_reload=True)
        self.model.conf = 0.5  # confidence threshold (0-1)
        # self.model.iou = 0.491  # NMS IoU threshold (0-1)
        self.image_size = 512
        self.results = {0: [], 1: []}
        self.squares = {0: 0, 1: 0}
        self.squares_conf = {0: [], 1: []}
        self.detects = None

    def show(self):
        return Image.fromarray(self.detects.render()[0])

    def save(self, filename):
        return Image.fromarray(self.detects.render()[0]).save(filename)

    @staticmethod
    def _check_inclusion(y, x, label):
        xmin = label[0]
        ymin = label[1]
        xmax = label[2]
        ymax = label[3]
        if (x < xmax) and (x > xmin) and (y < ymax) and (y > ymin):
            return True
        return False

    @staticmethod
    def _is_pil_image(img):
        return isinstance(img, Image.Image)

    def detect(self, image):
        if not self._is_pil_image(image):
            image = Image.fromarray(image)
        self.results = {0: [], 1: []}
        self.squares = {0: 0, 1: 0}
        self.squares_conf = {0: [], 1: []}
        self.detects = self.model(image, size=self.image_size)
        detection_info = self.detects.xyxy[0].tolist()
        for label in detection_info:  # label: xmin ymin xmax ymax  confidence classname
            self.results[int(label[5])].append(label)

        arr = np.array(image)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for classname, labels in self.results.items():
                    for label in labels:
                        if self._check_inclusion(i, j, label):
                            self.squares[classname] += 1
                            break

        for classname, labels in self.results.items():
            confs = [label[4] for label in labels]
            if confs:
                min_conf = min(confs)
                max_conf = max(confs)
            else:
                max_conf = min_conf = None
            self.squares_conf[classname] = [self.squares[classname], min_conf, max_conf]

        return self.squares_conf


detector = Detector(path_to_weights)


def detect_on_img(path_to_img, binary_mask):
    image = Image.open(path_to_img)
    detector.detect(image)
    detector.save(path_to_img)  # save image in same place, similar for video


def detect_on_video(path_to_video, binary_mask):
    return None
