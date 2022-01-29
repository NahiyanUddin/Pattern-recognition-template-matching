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


    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
        frames.append(image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    fps = 1000*(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / vidcap.get(cv2.CAP_PROP_POS_MSEC))


# Load a color image
# img = cv2.imread('frames/frame0.jpg',1)

#convert RGB image to Gray
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#load reference image
# ref_rgb = cv2.imread('reference.jpg',1)
# ref = cv2.cvtColor(ref_rgb,cv2.COLOR_BGR2GRAY)


def step1(img , ref ):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    refh, refw = ref.shape

    max = -1
    x,y = 0,0
    for i in range(h):
        for j in range(w):
            if i+refh <=h and j+refw <=w:
                val = np.sum(np.multiply(ref,gray[i:i+refh,j:j+refw]))


                if val >max :
                    max = val
                    x,y = i,j

    mFrame1 = cv2.rectangle(img, (y, x), (y + refw, x + refh), (0, 0, 255), 2)
    cv2.imwrite("matched frames (Logarithmic search)/frame%d.jpg" % 1, mFrame1)

    return x,y,mFrame1

# cv2.imshow('fig1',fig1)
# cv2.waitKey(0)

def logarithmicSearch():
    matchedFrames.clear()


    ref_rgb = cv2.imread('reference.jpg', 1)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)

    img = frames[0]

    refh, refw = ref.shape
    h, w , dim = img.shape


    x,y , mFrame1 = step1(img,ref)

    # print(fps)

    video = cv2.VideoWriter('Logarithmic search.avi',cv2.VideoWriter_fourcc(*'DIVX'),fps,(w,h))

    video.write(mFrame1)

    p = 4

    for fcount in range(1,frames.__len__()):
        img = frames[fcount]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p = 4
        upx = int(x - p / 2)
        lowx = int(x + p / 2)

        upy = int(y - p / 2)
        lowy = int(y + p / 2)

        max = -1
        x,y = 0, 0
        i = upx
        j = upy



        while(p!=0):
            for m in range(3):
                i = upx + m*int(p / 2)
                for n in range(3):
                    j = upy + n*int(p/2)
                    if i + refh <= h and j + refw <= w and i >=0 and j>=0 :
                        
                        val = np.sum(np.multiply(ref, gray[i:i + refh, j:j + refw]))
                        if val > max:
                            max = val
                            x, y = i, j


            p = (p//2)

        mFrame1 = cv2.rectangle(img, (y, x), (y + refw, x + refh), (0, 0, 255), 2)
        cv2.imwrite("matched frames (Logarithmic search)/frame%d.jpg" % fcount, mFrame1)
        video.write(mFrame1)

    cv2.destroyAllWindows()
    video.release()




movieSpiliting()
logarithmicSearch()





