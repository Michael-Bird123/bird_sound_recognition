import os

import librosa
import numpy as np
import pandas as pd
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AddGaussianSNR
from pandas import DataFrame
from pydub import AudioSegment
from torchlibrosa import Spectrogram, LogmelFilterBank


def argument_data(data_path, arguement_path):
    augment = Compose([
        AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=30, p=1),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
    ])

    bird_labels = os.listdir(data_path)

    for bird_label in bird_labels:
        bird_label_path = data_path + "\\" + bird_label
        save_label_path = arguement_path + "\\" + bird_label
        if not os.path.exists(save_label_path):
            os.makedirs(save_label_path)

        audio_names = os.listdir(bird_label_path)
        for audio_name in audio_names:
            if not audio_name.endswith(".mp3"):
                continue

            audio_path = bird_label_path + "\\" + audio_name

            data, fs = librosa.load(audio_path, sr=32000, mono=True)
            augmented_data = augment(samples=data, sample_rate=32000)
            augmented_data = augmented_data * 32767
            augmented_data = augmented_data.astype("int16")
            augmented_data = augmented_data.tobytes()

            save_audio_path = save_label_path + "\\" + audio_name[0:-4] + "_argument.mp3"
            print(save_audio_path)
            sound = AudioSegment(data=augmented_data, sample_width=2, frame_rate=fs, channels=1)
            sound.export(save_audio_path, format="mp3", bitrate="128k")


def extract_data(data_path, min_audio_count=2000):
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

    # 获取鸟的标签
    class_label = os.listdir(data_path)

    class_num = len(class_label)
    print(class_num)

    label_count = 0

    for i in range(class_num):
        classs_label_path = data_path + "\\" + class_label[i]
        audio_names = os.listdir(classs_label_path)
        audio_count = 0
        for audio_name in audio_names:
            if not audio_name.endswith(".mp3"):
                continue
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

            if audio_count > min_audio_count:
                break

        label_count += 1

    total_data = np.array(total_data)
    total_label = np.array(total_label)
    print(total_data.shape)
    print(total_label.shape)
    # 保存数据集
    np.save("./Bird_Dataset/total_data.npy", total_data)
    np.save("./Bird_Dataset/total_label.npy", total_label)


def Sort_class_label(data_path, min_audio_num=10):
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

    xlsx_data = {'鸟类标签': Bird_labels, '音频数': num_data}
    df = DataFrame(xlsx_data)
    df.to_excel("./Bird_Dataset/class_label.xlsx")

if __name__ == '__main__':
    data_path = "D:\\白云山鸟声训练数据集\\train_data"
    arguement_path = "D:\\白云山鸟声训练数据集\\argument_data"

    # argument_data(data_path, arguement_path)
    #extract_data(arguement_path, min_audio_count=2000)
    Sort_class_label(arguement_path, min_audio_num=10)
