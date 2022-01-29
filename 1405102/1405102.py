import cv2
import imageio
import numpy as np

eFrames = []
lFrames = []
hFrames = []

time = 0
framecount = 0
fps = 0.0

main_p = 4

matchedFrames = []

def movieSpiliting():
    global fps
    vidcap = cv2.VideoCapture('movie.mov')


    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
        eFrames.append(image)
        lFrames.append(image)
        hFrames.append(image)

        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    fps = 1000*(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / vidcap.get(cv2.CAP_PROP_POS_MSEC))


def step1(img , ref , color ):
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

    mFrame1 = cv2.rectangle(img, (y, x), (y + refw, x + refh), color, 2)
    cv2.imwrite("matched frames (Exhaustive search)/frame%d.jpg" % 1, mFrame1)

    return x,y,mFrame1


def exhaustiveSearch():
    print('...............................E X H A U S T I V E  S E A R C H..............................................')

    matchedFrames.clear()

    # frames = mainframes.copy()


    ref_rgb = cv2.imread('reference.jpg', 1)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)

    img = eFrames[0].copy()

    refh, refw = ref.shape
    h, w , dim = img.shape

    print('frame ',0)
    x,y , mFrame1 = step1(img,ref,(0,0,255))

    # print(fps)

    video = cv2.VideoWriter('output(Exhaustive search).avi',cv2.VideoWriter_fourcc(*'DIVX'),fps,(w,h))

    video.write(mFrame1)

    p = main_p

    tot_count = 0

    for fcount in range(1,eFrames.__len__()):
        # print('eFrame ', fcount)

        img = eFrames[fcount].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        upx = int(x - p / 2)
        lowx = int(x + p / 2)

        upy = int(y - p / 2)
        lowy = int(y + p / 2)

        max = -1
        x,y = 0, 0

        cnt = 0
        record_count = 0

        for i in range(upx,lowx+1):
            for j in range(upy,lowy+1):
                if i + refh <= h and j + refw <= w:
                    val = np.sum(np.multiply(ref, gray[i:i + refh, j:j + refw]))
                    cnt += 1
                    if val > max:
                        record_count = cnt
                        max = val
                        x, y = i, j
        tot_count += record_count

        mFrame1 = cv2.rectangle(img, (y, x), (y + refw, x + refh), (0, 0, 255), 2)
        cv2.imwrite("matched frames (Exhaustive search)/frame%d.jpg" % fcount, mFrame1)
        video.write(mFrame1)

    cv2.destroyAllWindows()
    video.release()

    print('exhaustive:',p,' ',tot_count/(eFrames.__len__()-1))


def logarithmicSearch():
    print('.............................L O G A R I T H M I C  S E A R C H.............................................')

    matchedFrames.clear()

    # frames = mainframes.copy()


    ref_rgb = cv2.imread('reference.jpg', 1)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)

    img = lFrames[0].copy()

    refh, refw = ref.shape
    h, w, dim = img.shape

    print('lFrame ', 0)
    x, y, mFrame1 = step1(img, ref,(0,255,0))

    # print(fps)

    video = cv2.VideoWriter('output(Logarithmic search).avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    video.write(mFrame1)

    # main_p = 4

    tot_count = 0
    for fcount in range(1, lFrames.__len__()):
        # print('lFrame ', fcount)

        img = lFrames[fcount].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        p = main_p
        upx = int(x - p / 2)
        lowx = int(x + p / 2)

        upy = int(y - p / 2)
        lowy = int(y + p / 2)

        max = -1
        x, y = 0, 0
        i = upx
        j = upy


        while (p != 0):
            cnt = 0
            rec_cnt = 0
            for m in range(3):
                i = upx + m * int(p / 2)
                for n in range(3):
                    j = upy + n * int(p / 2)
                    if i + refh <= h and j + refw <= w and i >= 0 and j >= 0:
                        cnt +=1
                        val = np.sum(np.multiply(ref, gray[i:i + refh, j:j + refw]))
                        if val > max:
                            rec_cnt = cnt
                            max = val
                            x, y = i, j
            tot_count+= rec_cnt


            p = (p // 2)

        mFrame1 = cv2.rectangle(img, (y, x), (y + refw, x + refh), (0, 255, 0), 2)
        cv2.imwrite("matched frames (Logarithmic search)/frame%d.jpg" % fcount, mFrame1)
        video.write(mFrame1)

    cv2.destroyAllWindows()
    video.release()
    print('logarithmic',p,' ',tot_count/(lFrames.__len__()-1))


def hierarcgySearch():
    print('................................H I E R A R C H Y  S E A R C H...............................................')
    matchedFrames.clear()

    # frames = mainframes.copy()


    ref_rgb = cv2.imread('reference.jpg', 1)
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_BGR2GRAY)


    img = hFrames[0].copy()

    refh, refw = ref.shape
    h, w, dim = img.shape

    print('frame ', 0)
    x, y, mFrame1 = step1(img, ref,(255,0,0))

    # print(fps)

    video = cv2.VideoWriter('output(Hierarchy search).avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    video.write(mFrame1)

    p = main_p

    ref1 = cv2.blur(ref,(2,2))

    ref1 = cv2.resize(ref1, (refw // 2, refh // 2))

    ref2 = cv2.blur(ref1, (2,2))

    ref2 = cv2.resize(ref2, (refw // 4, refh // 4))

    print(ref2.shape)

    tot_cnt = 0
    for fcount in range(1, hFrames.__len__()):
        # print('frame ',fcount)

        img0 = hFrames[fcount].copy()
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

        cnt = 0
        rec_cnt = 0
        for m in range(p+1):
            i = (x//4-p//2) + m
            for n in range(p+1):
                j = (y//4-p//2) + n
                if i + refh//4 <= h//4 and j + refw//4 <= w//4 and i >= 0 and j >= 0:
                    val = np.sum(np.multiply(ref2, gray2[i:i + refh//4, j:j + refw//4]))
                    cnt +=1

                    if val > max:
                        rec_cnt = cnt
                        max = val
                        x2, y2 = i, j
        tot_cnt+=rec_cnt
        # level 1

        x1 = x2*2
        y1 = y2*2

        cnt = 0
        rec_cnt = 0
        for m in range(p+1):
            i = (x2*2 - p//2) + m
            for n in range(p+1):
                j = (y2*2 - p//2) + n
                if i + refh//2 <= h//2 and j + refw//2 <= w//2 and i >= 0 and j >= 0:
                    cnt +=1
                    val = np.sum(np.multiply(ref1, gray1[i:i + refh // 2, j:j + refw // 2]))
                    if val > max:
                        rec_cnt = cnt
                        max = val
                        x1, y1 = i, j

        tot_cnt += rec_cnt
        # level 0

        x = x1 * 2
        y = y1 * 2
        cnt = 0
        rec_cnt = 0
        for m in range(p+1):
            i = (x1 * 2 - p//2) + m
            for n in range(p+1):
                j = (y1 * 2 - p//2) + n
                if i + refh <= h and j + refw <= w and i >= 0 and j >= 0:
                    cnt+=1
                    val = np.sum(np.multiply(ref, gray0[i:i + refh, j:j + refw]))
                    if val > max:
                        rec_cnt=cnt
                        max = val
                        x, y = i, j
        tot_cnt += rec_cnt
        mFrame1 = cv2.rectangle(img0, (y, x), (y + refw, x + refh), (255, 0, 0), 2)
        cv2.imwrite("matched frames (Hierarchy search)/frame%d.jpg" % fcount, mFrame1)
        video.write(mFrame1)

        # input()

    cv2.destroyAllWindows()
    video.release()
    print('hierarchy:',p,' ',tot_cnt/(hFrames.__len__()-1))




movieSpiliting()
exhaustiveSearch()
logarithmicSearch()
hierarcgySearch()


