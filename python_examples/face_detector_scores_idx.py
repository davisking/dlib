
from PIL import Image
import numpy as np
import dlib

img = np.array(Image.open('../examples/faces/2008_002506.jpg'))
detector = dlib.get_frontal_face_detector()

dets, scores, idx = detector.run(img, 1)

for i, d in enumerate(dets):
    print d, scores[i], idx[i]

