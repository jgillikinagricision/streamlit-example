import streamlit as st
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import threading
from zipfile import ZipFile

BOX_COLOR = (0, 255, 255) #Yellow
TF_LITE_MODEL = 'weeds_yolov5_v8.tflite'
CLASSES_OF_INTEREST = ['Marestail', 'Pigweed', 'Dogfennel', 'Curlydoc', 'MorningGlory', 'ItalianRyeGrass', 'Ragweed']
VIDEO_SOURCE = 0  # an integer number for an OpenCV supported camera or any video file

def draw_boxes(frame, boxes, threshold, labelmap, obj_of_int):
    (h, w) = frame.shape[:2] #np array shapes are h,w OpenCV uses w,h
    for (t, l, b, r), label_id, c in boxes:
        if c > threshold and label_id in obj_of_int:
            top_left, bottom_right = (int(l*w), int(t*h)), (int(r*w), int(b*h))
            cv2.rectangle(frame, top_left, bottom_right, BOX_COLOR, 2)
            cv2.putText(frame, f'{labelmap[int(label_id)]} {c:.4f}', top_left, cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOR)
    return frame

def resize_keep_aspect(img, req_shape):
    ratio = max((img.shape[1]/req_shape[0], img.shape[0]/req_shape[1]))
    new_h, new_w = int(img.shape[0]/ratio), int(img.shape[1]/ratio)
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img, 0, (req_shape[1]-new_h), 0, (req_shape[0]-new_w), cv2.BORDER_CONSTANT)
    return img, (req_shape[1]/new_h, req_shape[0]/new_w)

class CameraThread(threading.Thread):
    def __init__(self, name='CameraThread'):
        super().__init__(name=name, daemon=True)
        self.stop_event = False
        self.open_camera()
        self.setup_inference_engine()
        self._frame, self.results = np.zeros((300,300,3), dtype=np.uint8), [] #initial empty frame
        self.lock = threading.Lock()
        self.log_counter = 0

    def open_camera(self):
        self.webcam = cv2.VideoCapture(VIDEO_SOURCE)

    def setup_inference_engine(self):
        self.intp = tflite.Interpreter(model_path=TF_LITE_MODEL)
        self.intp.allocate_tensors()
        self.input_idx = self.intp.get_input_details()[0]['index']
        self.output_idxs = [i['index'] for i in self.intp.get_output_details()[:3]]

    def process_frame(self, img):
        _img, (rh, rw) = resize_keep_aspect(img, (300,300)) # cv2.resize(img, (300, 300))
        self.intp.set_tensor(self.input_idx, _img[np.newaxis, :])
        self.intp.invoke()
        boxes, label_id, conf = [self.intp.get_tensor(idx).squeeze() for idx in self.output_idxs]
        boxes = [(t*rh, l*rw, b*rh, r*rw) for (t, l, b, r) in boxes] # scale the coords back
        return list(zip(boxes, label_id, conf))

    def run(self):
        while not self.stop_event:
            ret, img = self.webcam.read()
            if not ret: #re-open camera if read fails. Useful for looping test videos
                self.open_camera()
                continue
            results = self.process_frame(img)
            with self.lock: self.results, self._frame = results, img.copy()
            self.log_frame()

    def log_frame(self):
        if len(self.results) > 0 and np.any([r[2] > 0.5 for r in self.results]):
            cv2.imwrite(f'detections_{self.log_counter:04d}.jpg', self._frame)
            self.log_counter += 1

    def stop(self): self.stop_event = True

    def read(self):
        with self.lock: return self._frame.copy(), self.results

@st.cache(allow_output_mutation=True)
def get_or_create_camera_thread():
    for th in threading.enumerate():
        if th.name == 'CameraThread':
            th.stop()
            th.join()
    cw = CameraThread('CameraThread')
    cw.start()
    return cw

@st.cache
def load_labelmap(filename='labelmap.txt'):
    return ZipFile(TF_LITE_MODEL).read(filename).decode('utf-8').splitlines()

camera = get_or_create_camera_thread()
label_map = load_labelmap()

st_frame = st.empty()
confidence_thresh = st.sidebar.slider('Confidence Threshold', value=0.5, key='threshold')
interested_classes = [label_map.index(l) for l in CLASSES_OF_INTEREST]

while True:
    frame, detections = camera.read()
    img = draw_boxes(frame, detections, confidence_thresh, label_map, interested_classes)
    st_frame.image(img, channels='BGR')
