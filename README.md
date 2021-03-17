### Summary


### Data preprocessing

Log-scaled mel-spectrograms is the modern standard way of the data representation in CNN-based audio scene classification.
Audio config parameters:  
```
sampling_rate = 32000
hop_length = 345 * 2
fmin = 20
fmax = sampling_rate // 2
n_mels = 128
n_fft = n_mels * 20
min_seconds = 10  #Seconds
```

### Augmentations 
Several augmentations were applied on spectrograms during the training stage using library [audiomentations](https://github.com/iver56/audiomentations). lists augmentation techniques:

```
        train_aug = aaug.Compose([
            aaug.AddGaussianNoise(p=0.5),
            aaug.AddGaussianSNR(p=0.5),
            aaug.Gain(p=0.5),
            aaug.AddBackgroundNoise(sounds_path = noise_dir, p=0.5)
            #aaug.Normalize(p=0.3)
            #aaug.Shift(p=0.2)
            #aaug.PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            #aaug.TimeStretch(p = 0.3)
       ])
```
### Training 

* 5 stratifiedkfold 
* Loss: BCE with auxiliary  
* Optimizer: AdamW with initial LR 0.0001 and weigth decauy Wd 1e-5 
* CosineAnnealingLR scheduler: factor 10  
* Use ImbalanceData Sampler for sampling curated and noisy data  
* Training with BCE on noisy samples with a high lwlrap score by previous models
* Mixed precision training


## References

[1] [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://github.com/qiuqiangkong/audioset_tagging_cnn)

[2] [Introduction to Sound Event Detection](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)

[3] [Audiomentations Python library for audio augmentation](https://github.com/iver56/audiomentations)

[4] [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)