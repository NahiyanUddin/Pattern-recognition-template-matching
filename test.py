import os
import cv2
import imageio
import numpy as np


fileList = os.scandir('frames/')

print(fileList)


def step1(img, ref):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    refh, refw = ref.shape

    max = -1
    x, y = 0, 0
    for i in range(h):
        for j in range(w):
            if i + refh <= h and j + refw <= w:
                val = np.sum(np.multiply(ref, gray[i:i + refh, j:j + refw]))

                if val > max:
                    max = val
                    x, y = i, j

    mFrame1 = cv2.rectangle(img, (y, x), (y + refw, x + refh), (0, 0, 255), 2)
    cv2.imwrite("matched frames (Hierarchy search)/frame%d.jpg" % 1, mFrame1)

    return x, y, mFrame1
