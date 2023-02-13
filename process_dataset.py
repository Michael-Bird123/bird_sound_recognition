import argparse
import os

import librosa
import numpy as np
from pandas import DataFrame
from pydub import AudioSegment


def sort_class_label(data_path, save_path, min_audio_num=10):
    if data_path == "":
        print("data path is null.")
        return -1
    class_labels = os.listdir(data_path)
    num_data = []
    Bird_labels = class_labels
    Bird_labels_copy = class_labels.copy()
    for class_label in Bird_labels_copy:
        label_path = data_path + "\\" + class_label
        audio_names = os.listdir(label_path)
        if len(audio_names) < min_audio_num:
            Bird_labels.remove(class_label)
        else:
            num_data.append(len(audio_names))

    xlsx_data = {'Bird_labels': Bird_labels, 'num_audio': num_data}
    if save_path == "":
        print("save path is null.")
        return -1
    df = DataFrame(xlsx_data)
    df.to_excel(save_path)


def cut_audio(data_path, save_path, cut_second=5, save_audio_extension=".mp3"):
    if data_path == "":
        print("data path is null.")
        return -1
    if save_audio_extension == ".mp3" or save_audio_extension == ".wav":
        print("save_audio_syntax is ", save_audio_extension)
    else:
        print("save audio syntax is error.")
        return -1
    for root, dirs, files in os.walk(data_path):
        # print("root:", root)
        # print("dirs:", dirs)
        # print("files", files)
        if files:
            # print(files)
            root_split = root.split(os.path.sep)
            # print(root_split)
            index = root_split.index(root_split[-1])
            save_data_path = save_path

            for i in range(index + 1, len(root_split)):
                save_data_path = save_data_path + "\\" + root_split[i]

            print(save_data_path)
            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)

            for audio_name in files:

                if audio_name.endswith(".wav") or audio_name.endswith(".mp3"):
                    count = 0
                    audio_path = root + "\\" + audio_name
                    print(audio_path)
                    data, fs = librosa.core.load(audio_path, sr=32000, mono=True)
                    data = data * 32767
                    length = len(data)
                    print(length)
                    if length <= fs * cut_second:
                        data = np.hstack((data, np.zeros(cut_second * fs - length)))
                        save_name = list(audio_name)
                        save_name.insert(-4, "_" + str(count))
                        save_name = ''.join(save_name)
                        save_name = save_name[0:-4] + save_audio_extension
                        save_path = save_data_path + "\\" + save_name

                        data = data.astype("int16")
                        data = data.tobytes()
                        sound = AudioSegment(data=data, sample_width=2, frame_rate=fs, channels=1)
                        if save_audio_extension == ".mp3":
                            sound.export(save_path, format="mp3", bitrate="128k")
                        elif save_audio_extension == ".wav":
                            sound.export(save_path, format="wav")

                        print("Saving file path：", save_path)
                    elif length > fs * cut_second:
                        cut_num = int(np.ceil(length / (fs * cut_second)))
                        data = np.array(data)
                        data = np.hstack((data, np.zeros(cut_num * fs * cut_second - length)))
                        for i in range(cut_num - 1):
                            cut_data = data[i * fs * cut_second:(i + 1) * fs * cut_second]
                            cut_data = np.array(cut_data)
                            save_name = list(audio_name)
                            save_name.insert(-4, "_" + str(count))
                            save_name = ''.join(save_name)
                            save_name = save_name[0:-4] + save_audio_extension
                            save_path = save_data_path + "\\" + save_name
                            cut_data = cut_data.astype("int16")
                            cut_data = cut_data.tobytes()

                            sound = AudioSegment(data=cut_data, sample_width=2, frame_rate=fs, channels=1)
                            if save_audio_extension == ".mp3":
                                sound.export(save_path, format="mp3", bitrate="128k")
                            elif save_audio_extension == ".wav":
                                sound.export(save_path, format="wav")
                            print("Saving file path：", save_path)
                            count += 1


def caesar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort_label', action='store_true', help="enable sort label")
    parser.add_argument('--cut_data', action='store_true', help="enable cut data")
    parser.add_argument('--data_path', default="")
    parser.add_argument('--save_path', default="")
    parser.add_argument('--cut_second', default=5, help="cut audio slice time")
    parser.add_argument('--save_audio_extension', default=".mp3", help="save audio syntax ,mp3 or wav")
    parser.add_argument('--min_audio_num', default=10, help="label audio min num")
    args = parser.parse_args()

    if args.sort_label:
        print("sorting label")
        sort_class_label(args.data_path, args.save_path, args.min_audio_num)
    elif args.cut_data:
        print("cutting data")
        cut_audio(args.data_path, args.save_path, args.cut_second, args.save_audio_extension)
    else:
        print("Nothing have done")


if __name__ == '__main__':
    caesar()
