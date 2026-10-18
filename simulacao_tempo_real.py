
import time
import numpy as np
import matplotlib

# Salva os gráficos em arquivo, sem depender do Tkinter
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# CONFIGURAÇÕES
# ==========================================

FS_EEG = 256
FS_EMG = 1000

DURACAO_JANELA = 5
INTERVALO = 0.1

rng = np.random.default_rng(42)

# ==========================================
# CONECTAR AO COPPELIASIM
# ==========================================

client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# Articulações monitoradas
nomes = [
    "index_base",
    "index_middle",
    "index_tip",
    "thumb_base",
    "thumb_tip"
]

juntas = {
    nome: sim.getObject("/" + nome)
    for nome in nomes
}

# ==========================================
# PREPARAR OS SINAIS
# ==========================================

tempo_eeg = np.arange(
    0, DURACAO_JANELA, 1 / FS_EEG
)

tempo_emg = np.arange(
    0, DURACAO_JANELA, 1 / FS_EMG
)

eeg = np.zeros(len(tempo_eeg))
emg_flexor = np.zeros(len(tempo_emg))
emg_extensor = np.zeros(len(tempo_emg))

# Filtro sEMG
sos_emg = butter(
    4,
    [20, 400],
    btype="bandpass",
    fs=FS_EMG,
    output="sos"
)

estado_flexor = np.zeros(
    (sos_emg.shape[0], 2)
)

estado_extensor = np.zeros(
    (sos_emg.shape[0], 2)
)

# ==========================================
# GERADORES DE SINAIS
# ==========================================

def gerar_eeg(t, atividade=0.5):
    """
    EEG sintético:
    Ritmo mu de 10 Hz
    Ritmo beta de 20 Hz
    Ruído de fundo
    """

    amplitude_mu = 15 * (1 - 0.6 * atividade)

    sinal = (
        amplitude_mu * np.sin(2 * np.pi * 10 * t)
        + 5 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 3, len(t))
    )

    return sinal


def gerar_semg(intensidade, n, estado):
    """
    Gera sEMG sintético filtrado.
    """

    ruido = rng.normal(0, 1, n)

    filtrado, novo_estado = sosfilt(
        sos_emg,
        ruido,
        zi=estado
    )

    sinal = (
        intensidade * filtrado
        + rng.normal(0, 0.02, n)
    )

    return sinal, novo_estado


def calcular_rms(sinal):
    return float(
        np.sqrt(np.mean(sinal ** 2))
    )


# ==========================================
# GRÁFICOS
# ==========================================

fig, axes = plt.subplots(
    4, 1,
    figsize=(12, 9)
)

fig.suptitle(
    "Monitor EEG + sEMG + InMoov"
)

def atualizar_graficos(tempo_atual, angulos):
    axes[0].clear()
    axes[0].plot(tempo_eeg, eeg)
    axes[0].set_title("EEG sintético")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].set_ylim(-35, 35)

    axes[1].clear()
    axes[1].plot(tempo_emg, emg_flexor)
    axes[1].set_title("sEMG Flexor")
    axes[1].set_ylabel("Amplitude (u.a.)")
    axes[1].set_ylim(-3, 3)

    axes[2].clear()
    axes[2].plot(tempo_emg, emg_extensor)
    axes[2].set_title("sEMG Extensor")
    axes[2].set_ylabel("Amplitude (u.a.)")
    axes[2].set_ylim(-3, 3)

    axes[3].clear()
    axes[3].bar(
        list(angulos.keys()),
        list(angulos.values())
    )
    axes[3].set_title(
        "Posições atuais das articulações"
    )
    axes[3].set_ylabel("Ângulo (graus)")
    axes[3].tick_params(
        axis="x",
        rotation=25
    )

    for ax in axes[:3]:
        ax.set_xlabel("Tempo na janela (s)")
        ax.grid(True)

    fig.tight_layout()

    # Arquivo atualizado durante a execução
    fig.savefig(
        "monitor_eeg_semg.png",
        dpi=100
    )


# ==========================================
# SIMULAÇÃO EM TEMPO REAL
# ==========================================

def main():
    global eeg
    global emg_flexor
    global emg_extensor
    global estado_flexor
    global estado_extensor

    print("Aguardando Play no CoppeliaSim...")

    while (
        sim.getSimulationState()
        == sim.simulation_stopped
    ):
        time.sleep(0.2)

    print("Iniciando monitoramento!")

    inicio = time.monotonic()
    ultima_imagem = inicio
    duracao = 30

    try:
        while time.monotonic() - inicio < duracao:

            if (
                sim.getSimulationState()
                == sim.simulation_stopped
            ):
                print("Simulação interrompida.")
                break

            t = time.monotonic() - inicio

            # Cenário sintético periódico:
            # flexão, extensão e repouso
            fase = int(t // 4) % 3

            if fase == 0:
                intensidade_flexor = 0.8
                intensidade_extensor = 0.02
                atividade_eeg = 1.0
                comando = "FECHAR"

            elif fase == 1:
                intensidade_flexor = 0.02
                intensidade_extensor = 0.8
                atividade_eeg = 1.0
                comando = "ABRIR"

            else:
                intensidade_flexor = 0.02
                intensidade_extensor = 0.02
                atividade_eeg = 0.0
                comando = "REPOUSO"

            n_eeg = round(FS_EEG * INTERVALO)
            n_emg = round(FS_EMG * INTERVALO)

            tempos_eeg = (
                t + np.arange(n_eeg) / FS_EEG
            )

            novo_eeg = gerar_eeg(
                tempos_eeg,
                atividade_eeg
            )

            novo_flexor, estado_flexor = gerar_semg(
                intensidade_flexor,
                n_emg,
                estado_flexor
            )

            novo_extensor, estado_extensor = gerar_semg(
                intensidade_extensor,
                n_emg,
                estado_extensor
            )

            # Atualizar janelas dos sinais
            eeg = np.roll(eeg, -n_eeg)
            eeg[-n_eeg:] = novo_eeg

            emg_flexor = np.roll(
                emg_flexor, -n_emg
            )
            emg_flexor[-n_emg:] = novo_flexor

            emg_extensor = np.roll(
                emg_extensor, -n_emg
            )
            emg_extensor[-n_emg:] = novo_extensor

            # Apenas ler posições do CoppeliaSim
            angulos = {}

            for nome, junta in juntas.items():
                angulos[nome] = np.degrees(
                    sim.getJointPosition(junta)
                )

            rms_flexor = calcular_rms(novo_flexor)
            rms_extensor = calcular_rms(novo_extensor)

            print(
                f"Tempo: {t:.1f}s | "
                f"Cenário: {comando} | "
                f"RMS flexor: {rms_flexor:.3f} | "
                f"RMS extensor: {rms_extensor:.3f}"
            )

            # Atualizar imagem a cada 1 segundo
            if time.monotonic() - ultima_imagem >= 1:
                atualizar_graficos(t, angulos)
                ultima_imagem = time.monotonic()

            time.sleep(INTERVALO)

    except KeyboardInterrupt:
        print("\nMonitoramento interrompido.")

    finally:
        plt.close(fig)
        print("Monitoramento finalizado.")


if __name__ == "__main__":
    main()
