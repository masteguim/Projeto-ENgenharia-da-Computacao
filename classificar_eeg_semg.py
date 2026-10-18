
import numpy as np
from scipy.signal import butter, sosfiltfilt

rng = np.random.default_rng(42)

FS_EEG = 256
FS_EMG = 1000


def gerar_eeg(atividade):
    """Gera um segundo de EEG sintético."""
    t = np.arange(FS_EEG) / FS_EEG

    amplitude_mu = 15 if atividade == "repouso" else 6

    eeg = (
        amplitude_mu * np.sin(2 * np.pi * 10 * t)
        + 5 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 3, len(t))
    )

    return eeg


def calcular_potencia_mu(eeg):
    """Calcula a potência média na banda de 8 a 13 Hz."""
    frequencias = np.fft.rfftfreq(len(eeg), 1 / FS_EEG)
    espectro = np.abs(np.fft.rfft(eeg)) ** 2

    banda = (frequencias >= 8) & (frequencias <= 13)

    return float(np.sum(espectro[banda]) / len(eeg) ** 2)


def gerar_semg(intensidade):
    """Gera um canal de sEMG sintético."""
    ruido = rng.normal(0, 1, FS_EMG)
    sos = butter(
        4, [20, 400],
        btype="bandpass",
        fs=FS_EMG,
        output="sos"
    )

    sinal = sosfiltfilt(sos, ruido)
    sinal /= max(np.sqrt(np.mean(sinal ** 2)), 1e-8)

    ruido_base = rng.normal(0, 0.02, FS_EMG)

    return intensidade * sinal + ruido_base


def calcular_ativacao(semg):
    rms = float(np.sqrt(np.mean(semg ** 2)))
    return float(np.clip((rms - 0.05) / 0.95, 0, 1))


def classificar(eeg, semg_flexor, semg_extensor):
    potencia_mu = calcular_potencia_mu(eeg)

    flexor = calcular_ativacao(semg_flexor)
    extensor = calcular_ativacao(semg_extensor)

    # EEG indica atividade motora
    if potencia_mu >= 30:
        comando = "REPOUSO"

    # sEMG indica a direção
    elif flexor - extensor > 0.15:
        comando = "FECHAR"

    elif extensor - flexor > 0.15:
        comando = "ABRIR"

    else:
        comando = "REPOUSO"

    return comando, potencia_mu, flexor, extensor


def testar(cenario, atividade, flexor, extensor):
    eeg = gerar_eeg(atividade)
    sinal_flexor = gerar_semg(flexor)
    sinal_extensor = gerar_semg(extensor)

    comando, potencia, af, ae = classificar(
        eeg, sinal_flexor, sinal_extensor
    )

    print(f"\nCenário: {cenario}")
    print(f"Potência mu: {potencia:.2f}")
    print(f"Ativação flexora: {af:.2f}")
    print(f"Ativação extensora: {ae:.2f}")
    print(f"Comando identificado: {comando}")

    return comando


if __name__ == "__main__":
    testar("Repouso", "repouso", 0.02, 0.02)
    testar("Fechar mão", "movimento", 0.9, 0.02)
    testar("Abrir mão", "movimento", 0.02, 0.9)
    testar("Repouso final", "repouso", 0.02, 0.02)
