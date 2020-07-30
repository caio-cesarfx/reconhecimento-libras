import cv2
import os
import pickle
from os.path import join, exists
import handsegment as hs
import argparse
from tqdm import tqdm

hc = []
gesture_folder = r"C:\Users\Caio Cesar\Desktop\TCC\\test_videos"
target_folder = r"C:\Users\Caio Cesar\Desktop\TCC\\test_frames"


def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not exists(majorData):
        os.makedirs(majorData)

    gesture_folder = os.path.abspath(gesture_folder)

    os.chdir(gesture_folder)
    gestures = os.listdir(os.getcwd())

    print("Diretório de origem contendo os gestos: %s" % (gesture_folder))
    print("Diretório de destino contendo frames: %s\n" % (majorData))

    for gesture in tqdm(gestures, unit='actions', ascii=True):
        gesture_path = os.path.join(gesture_folder, gesture)
        os.chdir(gesture_path)

        gesture_frames_path = os.path.join(majorData, gesture)
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if (os.path.isfile(video))]

        for video in tqdm(videos, unit='videos', ascii=True):
            name = os.path.abspath(video)
            cap = cv2.VideoCapture(name)  # Capturar input de vídeo
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lastFrame = None

            os.chdir(gesture_frames_path)
            count = 0

            # Suposição que só os 200 primeiros frames importam
            while count < 201:
                ret, frame = cap.read()  # extrai o frame
                if ret is False:
                    break
                framename = os.path.splitext(video)[0]
                framename = framename + "_frame_" + str(count) + ".jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])

                if not os.path.exists(framename):
                    frame = hs.handsegment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1

            # Repete último frame até ter 200 frames
            while count < 201:
                framename = os.path.splitext(video)[0]
                framename = framename + "_frame_" + str(count) + ".jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])
                if not os.path.exists(framename):
                    cv2.imwrite(framename, lastFrame)
                count += 1

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    os.chdir(rootPath)


convert(gesture_folder, target_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'Extract Individual Frames from gesture videos.')
    parser.add_argument('gesture_folder', help= 'Path to folder containing folder os videos of different gestures.')
    parser.add_argument('target_folder', help= 'Path to folder where extracted frames should be kept.')
    args = parser.parse_args()
    convert(args.gesture_folder, args.target_folder)
