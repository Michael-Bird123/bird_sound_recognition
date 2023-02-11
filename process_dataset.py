import argparse
import os
from pandas import DataFrame


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


def caesar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort_label', action='store_true', help="enable sort label")
    parser.add_argument('--data_path', default="")
    parser.add_argument('--save_path', default="")
    parser.add_argument('--min_audio_num', default=10, help="label audio min num")

    args = parser.parse_args()

    if args.sort_label:
        sort_class_label(args.data_path, args.save_path, args.min_audio_num)
        print("sorting label")

    else:
        print("Nothing have done")


if __name__ == '__main__':
    caesar()
