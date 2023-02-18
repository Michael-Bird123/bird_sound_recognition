import argparse
import os

import librosa
import numpy as np
import torch
from pandas import DataFrame
from pydub import AudioSegment
from torchlibrosa import Spectrogram, LogmelFilterBank


def sort_class_label(data_path, save_path):
    if data_path == "":
        print("data path is null.")
        return -1
    class_labels = os.listdir(data_path)
    num_data = []
    Bird_labels = class_labels
    Bird_labels_copy = class_labels.copy()
    for class_label in Bird_labels_copy:
        print(class_label)
        label_path = data_path + "\\" + class_label
        audio_names = os.listdir(label_path)
        num_data.append(len(audio_names))

    xlsx_data = {'Bird_labels': Bird_labels, 'num_audio': num_data}
    if save_path == "":
        print("save path is null.")
        return -1
    df = DataFrame(xlsx_data)
    df.to_excel(save_path + "/class_label.xlsx")


def mp3_to_wav_all_file(data_path, save_path):
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

            for i in range(index, len(root_split)):
                save_data_path = save_data_path + "\\" + root_split[i]
                #print(save_data_path)

            if not os.path.exists(save_data_path):
                os.makedirs(save_data_path)

            for audio_name in files:
                if not audio_name.endswith(".mp3"):
                    continue

                mp3_filename = root + "\\" + audio_name
                print(mp3_filename)
                wav_filename = save_data_path + "\\" + audio_name[0:-4] + ".wav"
                print(wav_filename)
                mp3_to_wav(mp3_filename, wav_filename)



def mp3_to_wav(mp3_filename, wav_filename):
    mp3_file = AudioSegment.from_mp3(file=mp3_filename)
    mp3_file.export(wav_filename, format="wav")


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

            for i in range(index, len(root_split)):
                save_data_path = save_data_path + "\\" + root_split[i]

            #print(save_data_path)
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
                    #print(length)
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


def extract_feature(data_path, save_dataset_path, min_audio_count=200):
    if data_path == "":
        print("data path is null.")
        return -1

    # spectrogram extractor
    spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320,
                                        win_length=1024, window='hann', center=True, pad_mode='reflect',
                                        freeze_parameters=True)

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024,
                                        n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None,
                                        freeze_parameters=True)

    total_data = []
    total_label = []

    # get bird labels
    class_label = os.listdir(data_path)

    class_num = len(class_label)
    print(class_num)

    label_count = 0

    for i in range(class_num):
        classs_label_path = data_path + "\\" + class_label[i]
        audio_names = os.listdir(classs_label_path)
        audio_count = 0
        for audio_name in audio_names:
            audio_path = classs_label_path + "\\" + audio_name
            print(audio_path)
            (wavedata, _) = librosa.core.load(audio_path, sr=32000, mono=True)
            wavedata = wavedata[None, :]  # (1, audio_length)
            wavedata = torch.Tensor(wavedata)
            spectrogram = spectrogram_extractor(wavedata)
            logmel = logmel_extractor(spectrogram)
            logmel_spectrogram = logmel[0].numpy()
            m_label = [0] * (class_num - 1)
            m_label.insert(label_count, 1)
            # print(m_label)
            total_data.append(logmel_spectrogram)
            total_label.append(m_label)
            audio_count += 1
            # make the balance data
            if audio_count > min_audio_count:
                break

        label_count += 1

    total_data = np.array(total_data)
    total_label = np.array(total_label)
    print(total_data.shape)
    print(total_label.shape)
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)

    # save the dataset
    np.save(save_dataset_path + "/total_data.npy", total_data)
    np.save(save_dataset_path + "/total_label.npy", total_label)


def caesar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort_label', action='store_true', help="enable sort label")
    parser.add_argument('--cut_data', action='store_true', help="enable cut data")
    parser.add_argument('--mp3_to_wav', action='store_true', help="enable mp3 to wav")
    parser.add_argument('--extract_feature', action='store_true', help="enable extract feature")
    parser.add_argument('--data_path', default="")
    parser.add_argument('--save_path', default="")
    parser.add_argument('--cut_second', default=5, help="cut audio slice time")
    parser.add_argument('--save_audio_syntax', default=".mp3", help="save audio syntax ,mp3 or wav")
    parser.add_argument('--min_audio_num', default=10, help="label audio min num")
    args = parser.parse_args()

    if args.sort_label + args.cut_data + args.mp3_to_wav > 1:
        print("error")
        return -1

    if args.sort_label:
        print("sorting label")
        sort_class_label(args.data_path, args.save_path)
    elif args.cut_data:
        print("cutting data")
        cut_audio(args.data_path, args.save_path, args.cut_second, args.save_audio_syntax)
    elif args.mp3_to_wav:
        print("mp3 to wav")
        mp3_to_wav_all_file(args.data_path, args.save_path)
    elif args.extract_feature:
        print("extract feature")
        extract_feature(args.data_path, args.save_path, min_audio_count=200)
    else:
        print("Nothing have done")


if __name__ == '__main__':
    caesar()
