import cv2
import imageio
import numpy as np

frames = []

time = 0
framecount = 0
fps = 0.0

matchedFrames = []


def movieSpiliting():
    global fps
    vidcap = cv2.VideoCapture('movie.mov')

    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
        frames.append(image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    fps = 1000 * (vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / vidcap.get(cv2.CAP_PROP_POS_MSEC))



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



def hierarcgySearch():
    matchedFrames.clear()

    ref_rgb = cv2.imread('reference.jpg', 1)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)


    img = frames[0]

    refh, refw = ref.shape
    h, w, dim = img.shape

    print('frame ', 0)
    x, y, mFrame1 = step1(img, ref)

    # print(fps)

    video = cv2.VideoWriter('Hierarchy search.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    video.write(mFrame1)

    p = 4

    ref1 = cv2.blur(ref,(2,2))

    ref1 = cv2.resize(ref1, (refw // 2, refh // 2))

    ref2 = cv2.blur(ref1, (2,2))

    ref2 = cv2.resize(ref2, (refw // 4, refh // 4))

    print(ref2.shape)

    for fcount in range(1, frames.__len__()):
        print('frame ',fcount)

        img0 = frames[fcount]
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        img1 = cv2.blur(img0,(2,2))

        img1 = cv2.resize(img1,(w//2,h//2))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


        x1 = x//2
        y1 = y//2

        img2 = cv2.blur(img1,(2,2))

        img2 = cv2.resize(img2, (w // 4, h // 4))
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        x2 = x//4
        y2 = y//4

        max = -1.0000

        # level 2

        for m in range(p+1):
            i = (x//4-p//2) + m
            for n in range(p+1):
                j = (y//4-p//2) + n
                if i + refh//4 <= h//4 and j + refw//4 <= w//4 and i >= 0 and j >= 0:
                    val = np.sum(np.multiply(ref2, gray2[i:i + refh//4, j:j + refw//4]))


                    if val > max:
                        max = val
                        x2, y2 = i, j

        # level 1

        x1 = x2*2
        y1 = y2*2
        for m in range(p+1):
            i = (x2*2 - p//2) + m
            for n in range(p+1):
                j = (y2*2 - p//2) + n
                if i + refh//2 <= h//2 and j + refw//2 <= w//2 and i >= 0 and j >= 0:

                    val = np.sum(np.multiply(ref1, gray1[i:i + refh // 2, j:j + refw // 2]))
                    if val > max:
                        max = val
                        x1, y1 = i, j


        # level 0

        x = x1 * 2
        y = y1 * 2
        for m in range(p+1):
            i = (x1 * 2 - p//2) + m
            for n in range(p+1):
                j = (y1 * 2 - p//2) + n
                if i + refh <= h and j + refw <= w and i >= 0 and j >= 0:

                    val = np.sum(np.multiply(ref, gray0[i:i + refh, j:j + refw]))
                    if val > max:
                        max = val
                        x, y = i, j

        mFrame1 = cv2.rectangle(img0, (y, x), (y + refw, x + refh), (0, 0, 255), 2)
        cv2.imwrite("matched frames (Hierarchy search)/frame%d.jpg" % fcount, mFrame1)
        video.write(mFrame1)

        # input()

    cv2.destroyAllWindows()
    video.release()


movieSpiliting()
hierarcgySearch()





