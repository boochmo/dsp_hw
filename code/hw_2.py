import numpy as np
import soundfile as sf
import torch
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


def adjust_length(signal, noise):
    return noise[: len(signal)]


def mixer(original, noise, snr_db):
    # получаем множитель, на который будем делить для заданного SNR
    factor = 10 ** (snr_db / 10)

    # обрезаем лишнее
    noise = adjust_length(original, noise)
    # коэффициент для сложения
    coef = np.sqrt((original**2).mean() / (factor * (noise**2).mean()))
    # миксуем
    mix = original + coef * noise
    return mix


if __name__ == "__main__":
    # получение данных
    orig, sr = sf.read("./data/hw_2/gt.wav")
    orig = torch.tensor(orig)
    noise, _ = sf.read("./data/hw_2/noise.wav")
    noise = noise[:, 1]

    # считаем метрики
    PESQ = PerceptualEvaluationSpeechQuality(16000, "wb")
    for snr_db in [-5, 0, 5, 10]:
        mix = mixer(orig, noise, snr_db)
        print(f"-=-=-=-=-=-=-=- SNRdB: {snr_db} -=-=-=-=-=-=-=-")
        print(
            "SNR: ",
            SignalNoiseRatio()(mix, orig).detach().numpy(),
        )
        print(
            "SDR: ",
            SignalDistortionRatio()(mix, orig).detach().numpy(),
        )
        print(
            "SI-SDR: ",
            ScaleInvariantSignalDistortionRatio()(mix, orig).detach().numpy(),
        )
        print(
            "PESQ: ",
            PESQ(mix, orig).detach().numpy(),
        )
        sf.write(f"./output/hw_2/snr_{snr_db}.wav", mix.clone().detach(), sr)
