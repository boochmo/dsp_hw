import scipy
import soundfile as sf
from scipy.fft import irfft, rfft
from scipy.io.wavfile import write


def deconvolve(x, y):
    if len(x) > len(y):
        x = x[: len(y)]
    else:
        y = y[: len(x)]
    X = rfft(x)
    Y = rfft(y)
    H = Y / X
    h = irfft(H)
    return h


if __name__ == "__main__":
    rec_noise, sr = sf.read("./data/hw_1/white_noise_rec.wav")
    rec_noise = rec_noise[:, 1]

    enh_noise, _ = sf.read("./output/hw_1/enh_white_noise.wav")
    enh_noise = enh_noise

    h = deconvolve(enh_noise, rec_noise)

    rec, _ = sf.read("./data/hw_1/record.wav")
    rec = rec[:, 1]

    restore = scipy.signal.convolve(h, rec, mode="full")[-len(rec) :]

    write("./output/hw_1/restore.wav", sr, restore)
