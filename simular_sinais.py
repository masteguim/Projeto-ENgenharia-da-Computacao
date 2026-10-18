import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

rng = np.random.default_rng(42)

# Frequências de amostragem
fs_eeg = 256
fs_emg = 1000
duracao = 6

t_eeg = np.arange(0, duracao, 1 / fs_eeg)
t_emg = np.arange(0, duracao, 1 / fs_emg)

# EEG sintético: ritmos mu (10 Hz) e beta (20 Hz)
# Amplitudes expressas em microvolts
envelope_mu = np.where(
    (t_eeg >= 2) & (t_eeg < 4), 0.4, 1.0
)

eeg = (
    15 * envelope_mu * np.sin(2 * np.pi * 10 * t_eeg)
    + 5 * np.sin(2 * np.pi * 20 * t_eeg)
    + rng.normal(0, 3, len(t_eeg))
)

# sEMG sintético: ruído de banda limitada
ruido = rng.normal(0, 1, len(t_emg))

sos = butter(
    4, [20, 400], btype="bandpass",
    fs=fs_emg, output="sos"
)

emg_base = sosfiltfilt(sos, ruido)

# Contração muscular entre 2 e 4 segundos
envelope_emg = np.where(
    (t_emg >= 2) & (t_emg < 4), 0.8, 0.05
)

emg = envelope_emg * emg_base

# Visualização
fig, axes = plt.subplots(2, 1, figsize=(11, 6))

axes[0].plot(t_eeg, eeg)
axes[0].set_title("EEG sintético")
axes[0].set_ylabel("Amplitude (µV)")
axes[0].grid(True)

axes[1].plot(t_emg, emg)
axes[1].set_title("sEMG sintético")
axes[1].set_ylabel("Amplitude (u.a.)")
axes[1].set_xlabel("Tempo (s)")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("sinais_eeg_semg.png", dpi=150)
plt.close()

print("Gráficos gerados com sucesso!")