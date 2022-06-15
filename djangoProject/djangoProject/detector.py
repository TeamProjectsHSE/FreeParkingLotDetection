import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

from djangoProject import settings

path_to_weights = settings.BASE_DIR.__str__() + R'\runs\train\yolov5x_results\weights\best.pt'


class Detector:

    def __init__(self, path_to_weights):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights, force_reload=True)
        self.model.conf = 0.566  # confidence threshold (0-1)
        #self.model.iou = 0.491  # NMS IoU threshold (0-1)
        self.image_size = 512
        self.detects = None
        self.labels = {0: 'free', 1: 'occupied'}
        self.colors = {0: (51, 255, 51), 1: (230, 44, 44)}
        self.out = None

        
    def show(self):
        return Image.fromarray(self.detects.render()[0])

    def save(self, filename):
        return self.out.save(filename)

    @staticmethod
    def _is_pil_image(img):
        return isinstance(img, Image.Image)

    def plot_one_box_PIL(self, boxes, im, line_thickness=3):
        draw = ImageDraw.Draw(im)
        for box, label in boxes:
            color = self.colors[label]
            lbl =  self.labels[label]
            draw.rectangle(box, width=line_thickness, outline=color)  # plot
            if lbl:
                font = ImageFont.load_default()
                txt_width, txt_height = font.getsize(lbl)
                draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=color)
                draw.text((box[0], box[1] - txt_height + 1), lbl, fill=(255, 255, 255))
        return im
  
    def detect(self, image, binary_mask):
        if not self._is_pil_image(image):
            image = Image.fromarray(image)
        self.detects = self.model(image, size=self.image_size)
        detection_info = self.detects.xywh[0].tolist()
        detection_info_plt = self.detects.xyxy[0].tolist()
        boxes = []
        for label in detection_info: #label: xmin ymin xmax ymax  confidence classname #class x_center y_center width height
            if binary_mask[int(label[1])][int(label[0])] != 1:
                xyxy = detection_info_plt[detection_info.index(label)][0:4]
                c = int(label[5])
                boxes.append((xyxy, c))
        self.out = self.plot_one_box_PIL(boxes, image)

detector = Detector(path_to_weights)    


def detect_on_img(path_to_img, binary_mask):
    image = Image.open(path_to_img)
    detector.detect(image, binary_mask)
    detector.save(path_to_img)  # save image in same place, similar for video


def detect_on_video(path_to_video, binary_mask):
    return None
