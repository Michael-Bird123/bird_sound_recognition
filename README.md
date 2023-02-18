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
python process_dataset.py --sort_label --data_path ./data/xeno-canto-dataset --save_path ./bird_dataset
```

After sorting out bird labels,we should cut the audio data into 5 second-slice(audio only mp3 or wav).

```
python process_dataset.py --cut_data --data_path ./wav_data/xeno-canto-dataset --save_path ./cut_data/xeno-canto-dataset --save_audio_syntax .mp3
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

```
python process_dataset.py --extract_feature --data_path ./cut_data/xeno-canto-dataset  --save_path ./bird_dataset
```

③Third step(Train the network)

```
python train.py --train_model --dataset_path ./bird_dataset  --save_model_path ./Save_Model/Cnn14_sed"
```

④The last step(inference the audio)

```
python inference.py --audio_path ./audio_you_want --model_path ./Save_Model/Cnn14_sed/best_loss.pkl"
```

That all!It is easy to use the code to train you bird net.

Remarks:
①make sure you bird dataset have enough data(more than 50).

②The audio data and dataset in this github are just the examples.

③you need to download the bird you want from the Xeno-canto.


