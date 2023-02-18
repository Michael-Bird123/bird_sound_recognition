import argparse
from Bird_Model import *
import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchlibrosa import Spectrogram, LogmelFilterBank


def inference(audio_path, model_path, fig_show=False):
    # spectrogram extractor
    spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320,
                                        win_length=1024, window='hann', center=True, pad_mode='reflect',
                                        freeze_parameters=True)

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024,
                                        n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None,
                                        freeze_parameters=True)

    frames_per_second = 32000 // 320

    df = pd.read_excel("./Bird_Dataset/class_label.xlsx")
    labels = df.iloc[:, 1].values
    class_num = len(labels)
    Model = eval("Cnn14_sed")
    model = Model(sample_rate=32000, window_size=1024,
                  hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                  classes_num=class_num)

    model.load_state_dict(torch.load(model_path))

    wav_data, fs = librosa.load(audio_path, sr=32000, mono=True)

    waveform = wav_data[None, :]  # (1, audio_length)
    waveform = torch.Tensor(waveform)
    spectrogram = spectrogram_extractor(waveform)
    logmel_data = logmel_extractor(spectrogram)

    transfer_model = model.cuda()
    logmel_data = logmel_data.cuda()

    # Forward
    with torch.no_grad():
        transfer_model.eval()
        batch_output_dict = transfer_model(logmel_data, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    # print('Sound event detection result (time_steps x classes_num): {}'.format(
    #    framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 3  # Show top results
    top_result_confidence = clipwise_output[sorted_indexes[0: top_k]]
    top_result_mat = framewise_output[:, sorted_indexes[0: top_k]]
    """(time_steps, top_k)"""
    top_result_label = np.array(labels)[sorted_indexes[0: top_k]]
    print(top_result_label)

    if fig_show:
        # Plot result
        stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=1024,
                                 hop_length=320, window='hann', center=True)
        frames_num = stft.shape[-1]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
        axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0: top_k]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')
        plt.show()


def caesar():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', action='store_true', help="enable inference")
    parser.add_argument('--audio_path', default="")
    parser.add_argument('--model_path', default="")
    parser.add_argument('--fig_show', default=True)
    args = parser.parse_args()

    print("inference audio")
    inference(args.audio_path, args.model_path, args.fig_show)


if __name__ == '__main__':
    caesar()
