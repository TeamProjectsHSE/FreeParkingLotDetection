import numpy as np
from django.shortcuts import render

from . import detector
from . import constance, settings
from .forms import UploadFileForm

import cv2


def handle_uploaded_file(f, filename):
    with open(filename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def get_full_context(request, context):
    return {**context}


def home(request):
    form = UploadFileForm()
    return render(request, 'pages/main.html',
                  get_full_context(request, {'PAGE_NAME': constance.HOME_PAGE_NAME, 'form': form}))


def load_img(request):
    if request.method == 'POST':
        filename = '/static/tmp.jpg'
        handle_uploaded_file(request.FILES['file'], settings.BASE_DIR.__str__() + filename)
        return render(request, 'pages/select_area.html',
                      get_full_context(request, {'PAGE_NAME': constance.HOME_PAGE_NAME, 'img_path': filename}))


def handle_img(request):
    filename = request.POST.get('img_path')
    img_path = settings.BASE_DIR.__str__() + filename
    binary_mask = np.zeros(cv2.imread(img_path).shape[:2])
    for name, rect in request.POST.items():
        if name[:4] == 'rect':
            x1, x2, y1, y2 = [int(float(i)) for i in rect.split(',')]
            for x in range(min(x1, x2), max(x1, x2)):
                for y in range(int(min(y1, y2)), max(y1, y2)):
                    binary_mask[x][y] = 1
    detector.detect_on_img(img_path, binary_mask)
    return render(request, 'pages/results.html',
                  get_full_context(request, {'PAGE_NAME': constance.HOME_PAGE_NAME, 'img_path': filename}))
