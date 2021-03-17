

class config(object):
    batch_size = 10
    sample_rate = 48000
    window_size = 2048
    hop_size = 1024
    mel_bins = 128
    fmax = sample_rate/2.0
    fmin = 40.0
    classes_num = 24

    data_dir = '../input/rfcx-species-audio-detection'
    noise_dir = "../input/random-noise-audio"
    sampler = True
    mixup = False
    