import os

import soundfile as sf
import torch
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


if __name__ == "__main__":
    files = ["snr_-5", "snr_0", "snr_5", "snr_10"]
    orig, sr = sf.read("./data/hw_2/gt.wav")
    orig = torch.tensor(orig)
    for file in files:
        query = f"deepFilter output/hw_2/{file}.wav --output-dir output/hw_3/"
        os.system(query)

        enh, _ = sf.read(f"./output/hw_3/{file}_DeepFilterNet3.wav")
        enh = torch.tensor(enh)

        PESQ = PerceptualEvaluationSpeechQuality(16000, "wb")

        ## аналогичная 2 заданию проблема - для пески и днсмоса надо ресемплить в 16кГц

        print(f"-=-=-=-=-=-=- file: {file}_DeepFilterNet3.wav -=-=-=-=-=-=-")
        print(
            "SNR: ",
            SignalNoiseRatio()(enh, orig).detach().numpy(),
        )
        print(
            "SDR: ",
            SignalDistortionRatio()(enh, orig).detach().numpy(),
        )
        print(
            "SI-SDR: ",
            ScaleInvariantSignalDistortionRatio()(enh, orig).detach().numpy(),
        )
        print(
            "PESQ: ",
            PESQ(enh, orig).detach().numpy(),
        )
