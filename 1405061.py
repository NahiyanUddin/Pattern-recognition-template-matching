import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

P_array = [4,8,12,16,20]
main_frames =[]

def plotGraph(x,y):
    plt.plot(x, y, markersize=7, color='blue', alpha=0.5)
    plt.xlabel('P')
    plt.ylabel('execution time')
    plt.title('logistic2D execution time vs P')
    plt.show()

def exhaustive_search(frame, ref_frame):
    refx = ref_frame.shape[0]
    refy = ref_frame.shape[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_padded = np.pad(ref_frame, ((0, frame.shape[0] - refx), (0, frame.shape[1] - refy)),\
                        'constant', constant_values=((0,0),(0,0)))
    
    cor = np.real(np.fft.ifft2((np.conj(np.fft.fft2(ref_padded))*np.fft.fft2(frame))/\
                             np.absolute(np.conj(np.fft.fft2(ref_padded))*np.fft.fft2(frame))))
    
    temp =  np.unravel_index(np.argmax(cor, axis=None), cor.shape)
    return int(temp[0] + refx/2), int(temp[1] + refy/2)

def exhaustive(vid_frames,ref_image,out):
    then = time.time()
    for i in range(len(vid_frames)):
        x, y = exhaustive_search(vid_frames[i], ref_image)
        frame = cv2.rectangle(vid_frames[i], (int(y - ref_image.shape[1] / 2), int(x - ref_image.shape[0] / 2)), \
                              (int(y + ref_image.shape[1] / 2), int(x + ref_image.shape[0] / 2)), (0, 0, 255), 3)
        out.write(frame)
    return time.time()-then
def partHierarchical(image,ref_image,div,M,N,ix,iy,p):

    dir_xy = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]

    max_corr = -999999
    max_x = 0
    max_y = 0

    for j in range(len(dir_xy)):
        tx = int(ix + p*dir_xy[j][0]//div)
        ty = int(iy + p*dir_xy[j][1]//div)
        part_image = image[tx:tx + M // div, ty:ty + N // div]

        print("ref ",ref_image.shape)
        print("part ",part_image.shape)

        part_image = cv2.resize(part_image, (M // div, N // div))
        ref_image = cv2.resize(ref_image, (M // div, N // div))


        corr = np.sum(np.multiply(part_image, ref_image))
        if (max_corr < corr):
            max_corr = corr
            max_x = tx
            max_y = ty

    ix = max_x
    iy = max_y

    return ix,iy







def hierarchical(vid_frames, ref_image,out,p):

    timeList = []


    #print("P is ",P)
    #vid_frames = updateVidFrames(vid_frames)

    test = vid_frames[0]
    #print("shape of frame ", test.shape)

    RX, RY, _ = ref_image.shape

    ref_image1 = cv2.blur(ref_image, (5, 5))
    ref_image1 = cv2.resize(ref_image1, (RY // 2, RX // 2))
    ref_image2 = cv2.blur(ref_image1, (5, 5))
    ref_image2 = cv2.resize(ref_image2, (RY // 4, RX // 4))

    print("RX RY ",RX," ",RY)

    then = time.time()

    fx, fy = exhaustive_search(vid_frames[0], ref_image)
    #print(str(0) + " : ( " + str(lx) + " , " + str(ly) + " )")
    write_frame = cv2.rectangle(main_frames[0], (fy, fx), (int(fy - ref_image.shape[1]), int(fx - ref_image.shape[0])), (255, 0, 0), 3)
    out.write(write_frame)

    ix = fx // 4
    iy = fy // 4

    for i in range(len(vid_frames)-2):
        frame = vid_frames[i+1]
        IX, IY, _ = frame.shape

        #print("IX IY ",IX," ",IY)

        frame1 = cv2.blur(frame, (5, 5))
        frame1 = cv2.resize(frame1, (IY // 2, IX // 2))
        frame2 = cv2.blur(frame1, (5, 5))
        frame2 = cv2.resize(frame2, (IY // 4, IX // 4))

        #print("ix,iy ",ix," ",iy)
        ix,iy = partHierarchical(frame2,ref_image2,4,RX,RY,ix,iy,p)

        ix = ix * 2
        iy = iy * 2
        ix,iy = partHierarchical(frame1,ref_image1,2,RX,RY,ix,iy,p)
        ix = ix * 2
        iy = iy * 2
        #print("success")
        ix,iy = partHierarchical(frame, ref_image, 1, RX, RY, ix, iy,p)
        #print("ip,iq ", ix, " ", iy)


        #print(str(i+1) + " : ( " + str(ux) + " , " + str(uy) + " )")
        write_frame = cv2.rectangle(main_frames[i+1], (iy, ix), (iy + ref_image.shape[1], ix + ref_image.shape[0]), (255, 0, 0), 3)
        out.write(write_frame)
        #cv2.imwrite("frame%d.jpg" % (i+1), frame)

        ix = ix // 4
        iy = iy // 4
    now = time.time()

    print("executing time ",now-then)
    return now-then

def partLog(image,ref_image,mult,M,N,ix,iy):

    dir_xy = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]

    max_corr = -999999
    max_x = 0
    max_y = 0

    for j in range(len(dir_xy)):
        tx = int(ix + mult*dir_xy[j][0])
        ty = int(iy + mult*dir_xy[j][1])
        part_image = image[tx:tx + M , ty:ty + N ]

        part_image = cv2.resize(part_image, (M, N ))
        ref_image = cv2.resize(ref_image, (M,N))
                # print(resize_image2.shape)
                # print(croped_image.shape)
        corr = np.sum(np.multiply(part_image, ref_image))
        if (max_corr < corr):
            max_corr = corr
            max_x = tx
            max_y = ty

    ix = max_x
    iy = max_y

    return ix,iy


def logistic2d(vid_frames, ref_image, tP,out):
    RX, RY, K = ref_image.shape

    then = time.time()
    fx, fy = exhaustive_search(vid_frames[0], ref_image)
    #print(str(0) + " : ( " + str(lx) + " , " + str(ly) + " )")
    write_frame = cv2.rectangle(main_frames[0], (fy, fx), (fy + ref_image.shape[1], fx + ref_image.shape[0]), (255, 0, 0), 3)
    out.write(write_frame)

    print(" shape ",vid_frames[0].shape)
    print(" ref shape ",RX," ",RY)

    ix = fx
    iy = fy


    dir_xy = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]

    for i in range(len(vid_frames)-2):
        frame = vid_frames[i+1]
        p = tP
        while(p>=1):
            ix,iy = partLog(frame,ref_image,p,RX,RY,ix,iy)
            p = p/2

        #print(str(i+1) + " : ( " + str(ux) + " , " + str(uy) + " )")
        write_frame = cv2.rectangle(main_frames[i+1], (iy, ix), (iy + ref_image.shape[1], ix + ref_image.shape[0]), (255, 0, 0), 3)
        out.write(write_frame)
    now = time.time()
    return now-then
        #cv2.imwrite("frame%d.jpg" % (i+1), image)



inp = cv2.VideoCapture('movie.mov')
ref = cv2.imread('reference.jpg')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputEx.mov',fourcc, inp.get(cv2.CAP_PROP_FPS),\
                      (int(inp.get(cv2.CAP_PROP_FRAME_WIDTH)),int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if (inp.isOpened()== False):
    print("Error opening video stream or file")
    exit(0)

vid_images = []
while inp.isOpened() :
    val,frame = inp.read()
    if val :
        vid_images.append(frame)
        #print("getting input")
    else :
        break

print("test ",ref.shape)
main_frames = vid_images.copy()


e_time = exhaustive(vid_images,ref,out)
print("e_time ",e_time)


array = []
#
# for i in range(len(P_array)):
#     #print("i ",i-1)
#     p = P_array[i]
#     #test_images = main_frames.copy()
#     #vid_frames, ref = updateAll(vid_images, ref,p)
#
#     array.append(hierarchical(vid_images, ref, out,p))
#     #array.append(logistic2d(vid_images, ref, p,out))
#

#logistic2d(vid_images,ref,P,out)
#exhaustive(vid_images,ref,out)


inp.release()
out.release()


#print(array)

#plotGraph(P_array,array)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           1405061/                                                                                            000775  001750  001750  00000000000 13421163137 011645  5                                                                                                    ustar 00reza                            reza                            000000  000000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         