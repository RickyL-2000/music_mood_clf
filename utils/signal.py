import numpy as np
import pyworld as pw
import parselmouth
from utils import utils
import scipy.signal
import librosa
import matplotlib.pyplot as plt
import warnings

def frame2sec(x, fs=22050, hop_length=128):
    return x * hop_length / fs

def sec2frame(x, fs=22050, hop_length=128):
    return np.ceil(x * fs / hop_length)

def get_f0(y, fs, method='parselmouth', f0_min=100, f0_max=800, cutoff=True, hop_size=256):
    if len(y) <= 0:
        print('ERROR in get_f0: The size of audio sequence should be positive. Skip error...')
        return None, None
    if method == 'world':
        try:
            f0, t = pw.dio(y, fs)
            if cutoff:
                f0[f0 < f0_min] = 0.0
                f0[f0 > f0_max] = 0.0
            return f0, t
        except Exception as err:
            print('The following exception occurs:')
            print(err)
            print('continue...')
            return None, None
    elif method == 'parselmouth':
        try:
            time_step = hop_size / fs
            snd = parselmouth.Sound(y, sampling_frequency=fs)
            f0 = snd.to_pitch(time_step=time_step, pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            t = np.arange(0.0, time_step * len(f0), time_step)
            t = t[:len(f0)]  # 可能会有浮点数不精确导致的长度不一
            if cutoff:
                f0[f0 < f0_min] = 0.0
                f0[f0 > f0_max] = 0.0
            return f0, t
        except Exception as err:
            print('The following exception occurs:')
            print(err)
            print('continue...')
            return None, None
    else:
        print('ERROR in get_f0: please specify the method correctly')
        return None, None

def extract_vocal(dst_path, src_path, verbose=True):
    if type(src_path) == list:
        ret = common.exe_cmd(
            'spleeter separate -o {} '.format(dst_path) +
            ' '.join(src_path),
            verbose
        )
        return ret
    elif type(src_path) == str:
        ret = common.exe_cmd(
            'spleeter separate -o {} {}'.format(dst_path, src_path), verbose
        )
        return ret
    else:
        print('ERROR: wrong param')
        return None

def note2freq(note, a4=440):
    if type(note) is np.ndarray:
        ret = np.zeros(shape=note.shape)
        tmp = a4 * 2 ** ((note - 69) / 12)
        ret[~np.isclose(note, -1)] = tmp[~np.isclose(note, -1)]
        return ret
    if np.isclose(note, -1):
        return 0.0
    return a4 * 2 ** ((note - 69) / 12)

def freq2note(freq, a4=440):
    if type(freq) is np.ndarray:
        ret = -np.ones(shape=freq.shape)
        tmp = np.log2(freq/a4 + 1e-6) * 12 + 69
        ret[~np.isclose(freq, 0.0)] = tmp[~np.isclose(freq, 0.0)]
        return ret
    if np.isclose(freq, 0.0):
        return -1
    return np.log2(freq / a4) * 12 + 69

def audio_filter(y, fs=22050, lowcut=100.0, highcut=800.0, order=6):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.0)
    high = min(highcut / nyq, 1.0)
    if np.isclose(low, 0.0) and high != np.inf:
        b, a = scipy.signal.butter(order, high, btype='low', output='ba')
    elif high == np.inf and not np.isclose(low, 0.0):
        b, a = scipy.signal.butter(order, low, btype='high', output='ba')
    elif not np.isclose(low, 0.0) and high != np.inf:
        b, a = scipy.signal.butter(order, [low, high], btype='band', output='ba')
    else:
        return y
    return scipy.signal.filtfilt(b, a, y).copy(order='C')

def mean_filt(y, k, dilation=0):
    assert dilation >= 0
    if dilation == 0:
        kernel = np.ones(shape=k) / k
    else:
        kernel = np.zeros(shape=k + (k-1) * dilation)
        for i in range(0, len(kernel), dilation+1):
            kernel[i] = 1/k
        k = k + (k-1) * dilation
    # _y = np.append([0.0] * (k//2), y)
    # _y = np.append(y, [0.0] * (k//2))
    ret = np.zeros(shape=y.shape)
    for i in range(k//2, y.shape[0] - k//2):
        ret[i] = np.sum(kernel * y[i-k//2: i+k//2+1])
    return ret

def load_audio(f_path, fs=22050, filtered=False, lowcut=0.0, highcut=800.0, filter_warnings=True):
    if not filter_warnings:
        y, fs = librosa.load(f_path, dtype=float, sr=fs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, fs = librosa.load(f_path, dtype=float, sr=fs)
    if len(y.shape) > 1 and y.shape[0] > 1:     # multi-channel
        y = np.mean(y, axis=1)
    if filtered:
        y = audio_filter(y, fs, lowcut, highcut)
    return y, fs

def plot_note_freq_mel(y, f0, freq, t, fs, start_time=0.0, title='', mel=False, mel_dB=True, filter0=True):
    """
    一个简易的画图函数。正经的函数在eval/plot.py里

    :param y: raw seq
    :param f0: f0 seq from raw seq
    :param freq: freq seq from note scripts
    :param t: time seq
    :param fs: sr
    :param start_time: the start time of the seq
    :param title: title of the plot
    :param mel: draw the mel spectrogram as background
    :param mel_dB: draw the mel spectrogram in dB
    :param filter0: filter the 0 points
    :return:
    """
    plt.figure(1)
    if mel:
        mel_spect = librosa.feature.melspectrogram(y=y, sr=fs)
        if mel_dB:
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        _freq = np.copy(freq)
        _f0 = np.copy(f0)
        _freq = _freq * (mel_spect.shape[0] / (np.max(_f0) + 50))
        _f0 = _f0 * (mel_spect.shape[0] / (np.max(_f0) + 50))
        if filter0:
            _freq[np.isclose(freq, 0.0)] = np.nan
            _f0[np.isclose(f0, 0.0)] = np.nan
        _t = t * (mel_spect.shape[1] / t[-1])
        plt.pcolor(mel_spect, alpha=0.5)
        plt.plot(_t, _freq, label='pitch', linewidth=2, color='k')
        plt.plot(_t, _f0, label='f0', color='r', alpha=0.6)
        plt.xticks(np.linspace(0, mel_spect.shape[1], 6), np.around(np.linspace(start_time, start_time + len(y) / fs, 6), 2))
        plt.yticks(np.linspace(0, mel_spect.shape[0], 6), np.around(np.linspace(0, np.max(f0) + 50, 6), 2))
    else:
        _freq = np.copy(freq)
        _f0 = np.copy(f0)
        if filter0:
            _freq[np.isclose(freq, 0.0)] = np.nan
            _f0[np.isclose(f0, 0.0)] = np.nan
        plt.plot(t, _freq, label='pitch', linewidth=2)
        plt.plot(t, _f0, label='f0', alpha=0.6)

    plt.xlabel('time')
    plt.ylabel('freq')
    plt.legend()
    plt.title(title)
    plt.show()

def wav2mp3(wav_path, mp3_path='', fs=22050, del_wav=False, verbose=True):
    if mp3_path == '':
        mp3_path = wav_path[:-4] + '.mp3'
    common.exe_cmd(
        f'ffmpeg -threads 1 -loglevel error -i "{wav_path}" -vn -ar {fs} -ac 1 -b:a 192k -y -hide_banner "{mp3_path}"',
        verbose=verbose
    )
    if del_wav:
        common.exe_cmd(f'rm -f "{wav_path}"')

def audio_len(path):
    """
    :return: 返回 microseconds
    """

    def _time2sec(time_str):
        start = 0
        end = time_str.find(':')
        ret = int(time_str[start: end]) * 3600
        start = end + 1
        end = time_str.find(':', start)
        ret += int(time_str[start: end]) * 60
        start = end + 1
        ret = float(time_str[start:]) + float(ret)
        return ret
    info = common.exe_cmd("ffprobe " + path, verbose=False)
    # pattern = re.compile("Duration: (.*?):(.*?):(.*?), start")
    # matcher = pattern.match(info[0].decode())
    text = info[0].decode()
    time_str = text[text.find("Duration") + 10: text.find(", start")]
    length = _time2sec(time_str)
    return length * 1000
