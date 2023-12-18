import copy
import os

import numpy as np
import soundfile as sf
from scipy.fft import irfft, rfft
from scipy.io.wavfile import write


if __name__ == "__main__":
    # считываем аудио
    orig, sr = sf.read("./data/hw_1/sweeper_gt.wav")
    orig = orig[:, 1]

    rec, _ = sf.read("./data/hw_1/sweeper_rec.wav")
    rec = rec[:, 1]
    rec = rec[: len(orig)]

    # получаем спектр
    orig_sp = np.abs(rfft(orig))
    rec_sp = np.abs(rfft(rec))

    # бъём на бины оригинальное аудио
    STEP = len(rec_sp) // 32
    orig_bins = [orig_sp[(i - 1) * STEP : i * STEP].mean() for i in range(1, 32)]
    orig_bins.append(orig_sp[-STEP:].mean())
    orig_bins = np.abs(orig_bins)     ## лишнее, уже делали же абс на спектре

    # бъём на бины записанное аудио
    rec_bins = [rec_sp[(i - 1) * STEP : i * STEP].mean() for i in range(1, 32)]
    rec_bins.append(rec_sp[-STEP:].mean())
    rec_bins = np.abs(rec_bins)

    # получаем наши гейны
    gains = orig_bins / rec_bins

    # по факту частот выше 16 кГц нет, там будет просто ноль,
    # поэтому их надо выкинуть
    for i, gain in enumerate(gains):
        if gain < 0.001:
            gains[i] = 1

    ## формулировка не очень совпадает с кодом, кажется что просто выкидываем частоты,
    ## хотя по факту ты просто гейны корректируешь в адекватные значения
    ## (что в целом тоже не очень честно, так как мы же стараемся колонку скорректировать,
    ## раз она не воспроизводит высокие - надо их корректировать, чтобы хотя бы попыталась)

    # считываем шум
    noise_orig, _ = sf.read("./data/hw_1/white_noise_gt.wav")
    noise_orig = noise_orig[:, 1]

    # получаем спетр шума
    noise_orig_sp = rfft(noise_orig)

    STEP_NOISE = len(noise_orig_sp) // 32

    # корректируем спектр
    enh_noise_sp = copy.deepcopy(noise_orig_sp)
    for i, gain in enumerate(gains):
        enh_noise_sp[i * STEP_NOISE : (i + 1) * STEP_NOISE] *= gain

    # переходим обратно во временную обласит
    enh_noise = irfft(enh_noise_sp)
    os.system("mkdir output/hw_1")
    write("./output/hw_1/enh_white_noise.wav", sr, enh_noise)


    ## хотелось бы какие-нибудь визуализации графиками, типа как выглядел сигнал до, как после, как Фурье выглядит и прочее
