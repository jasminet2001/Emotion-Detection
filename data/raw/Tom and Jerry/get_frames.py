import cv2
import math

videoFilePath = 'videos/Train Tom and jerry.mp4'
saveFolder = 'frames/train_frames/'
# saveFolder = 'frames/test_frames/'

cap = cv2.VideoCapture(videoFilePath)
frameRate = cap.get(5)

i = 0
count = 0

while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    count += math.floor(frameRate)
    print(count)

    ret, frame = cap.read()
    if (ret != True) or cv2.waitKey(1) & 0xFF == ord('q') or count >= 8912:
        break

    cv2.imwrite(saveFolder + "frame" + str(i) + ".jpg", frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
