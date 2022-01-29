import cv2

frames = []

def movieSpiliting():
    vidcap = cv2.VideoCapture('movie.mov')
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
      frames.append(image)
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1