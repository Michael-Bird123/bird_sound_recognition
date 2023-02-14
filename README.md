# bird_sound_recognition

In this place,you can easily to learn how to use deep learning to achieve bird_sound_recognition.

①First Step(download bird sound)
download the bird sound data from the Xeno-Canto. www.xeno-canto.org.

you can use the xcdl.py to download the bird sounds that you want.

```
python xcdl.py apus
```
apus is a bird name.

②Second Step(process bird dataset)

If you have dowmload you bird sounds that you want,now we should sort out you bird labels.


```
python process_dataset.py --sort_label --data_path ./data/xeno-canto-dataset --save_path ./bird_dataset/class_label.xlsx
```

After sorting out bird labels,we should cut the audio data into 5 second-slice(audio only mp3 or wav).

```
python process_dataset.py --cut_data --data_path ./data/xeno-canto-dataset --save_path ./cut_data/xeno-canto-dataset --save_audio_syntax .mp3
```

if cutting mp3 have some problem,you may tranform mp3 to wav.

```
python process_dataset.py --mp3_to_wav --data_path ./data/xeno-canto-dataset --save_path ./wav_data/xeno-canto-dataset 
```

Now you can cut data again.

```
python process_dataset.py --cut_data --data_path ./wav_data/xeno-canto-dataset --save_path ./cut_data/xeno-canto-dataset 
```

After cut data,we should extract the mel-specgram for train.

