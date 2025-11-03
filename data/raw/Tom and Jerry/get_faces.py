import os
import cv2
import numpy as np

video_path = 'videos/train/'
# video_path = 'videos/test/'


# gets the import location for videos (dataset)
def get_dataset(path=video_path):
    videos = os.listdir(os.path.dirname(path))
    return videos


# detect character by using a custom trained haar cascade for each character
def detect(character, video, show_video=True, interval_seconds=3):
    cap = cv2.VideoCapture(video_path + video)
    face_cascade = cv2.CascadeClassifier(character['cascade'])
    results_path = os.path.join('faces/train/' + character['name'])
    # results_path = os.path.join('faces/test/' + character['name'])

    # make a folder in results for our recognised faces
    if not os.path.exists(results_path) and character['save'] == True:
        os.mkdir(results_path)

    # determine frame interval (based on FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_count = 0

    while True:
        # grab a frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # process only every Nth frame
        if frame_count % frame_interval != 0:
            continue

        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.10,
            minNeighbors=40 if character['name'] == "Tom" else 20,
            minSize=(24, 24),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # loop over detected faces
        for (x, y, w, h) in faces:
            # setup region of interest (ROI) for the captured face
            roi = frame[y:y + h, x:x + w]

            frame_number = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            # write detected face to disk
            if character['save']:
                cv2.imwrite(os.path.join(results_path, f'frame_{frame_number}.png'), roi)

            if show_video:
                # display detection box for visual purposes
                cv2.rectangle(frame, (x, y), (x + w, y + h), character['detect_color'], 2)
                cv2.putText(frame, character['name'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                print(f"detected {character['name']} at frame {frame_number}")

        if show_video:
            # display our image
            try:
                display_frame = cv2.resize(frame, (960, 540))
                cv2.imshow('frame', display_frame)
            except:
                break

            # quit or (next video) on esc
            esc = cv2.waitKey(30) & 0xff
            if esc == 27:
                break

    # destroy & release resources
    cv2.destroyAllWindows()
    cap.release()


# process all our videos
def process(character):
    videos = get_dataset()
    print('number of videos: ' + str(len(videos)))

    for video in enumerate(videos):
        # dump frames and save to disk each character
        print('attempting to detect ' + character['name'])

        # detect our character
        detect(character, video[1], show_video=True)


def main():
    # prepare our results folder
    if not os.path.exists('faces'):
        os.mkdir('faces')

    # process all our videos to detect Tom & Jerry
    characters = [
        {
            'name': "Tom",
            'detect_color': (165, 91, 0),
            'save': True,
            'cascade': 'haar_cascades/tom.xml'
        },
        {
            'name': "Jerry",
            'detect_color': (165, 100, 0),
            'save': True,
            'cascade': 'haar_cascades/jerry.xml'
        }
    ]

    # process characters...
    [process(character) for character in characters]
    print('done')


if __name__ == '__main__':
    main()
