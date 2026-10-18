
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ======================================
# CONFIGURACOES
# ======================================

FS_EEG = 256
FS_EMG = 1000

INTERVALO = 0.2
DURACAO = 24

rng = np.random.default_rng(42)

client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# Articulacoes para acompanhar a resposta
juntas = {
    "base": sim.getObject("/index_base"),
    "meio": sim.getObject("/index_middle"),
    "ponta": sim.getObject("/index_tip")
}

filtro_emg = butter(
    4, [20, 400],
    btype="bandpass",
    fs=FS_EMG,
    output="sos"
)

# Historicos
tempos_eeg = []
valores_eeg = []

tempos_emg = []
valores_flexor = []
valores_extensor = []

tempos_controle = []
comandos = []
angulos_indicador = []
potencias_mu = []
rms_flexor = []
rms_extensor = []

# ======================================
# GERACAO EEG
# ======================================

def gerar_eeg(atividade, tempo):
    t = tempo + np.arange(FS_EEG) / FS_EEG

    # Ritmo mu: 10 Hz
    amplitude_mu = 6 if atividade else 15

    sinal = (
        amplitude_mu * np.sin(2 * np.pi * 10 * t)
        + 5 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 3, len(t))
    )

    return sinal


# ======================================
# GERACAO sEMG
# ======================================

def gerar_semg(intensidade):
    ruido = rng.normal(0, 1, FS_EMG)

    muscular = sosfiltfilt(
        filtro_emg, ruido
    )

    # Normalizacao RMS
    rms = np.sqrt(np.mean(muscular ** 2))
    muscular /= max(rms, 1e-8)

    fundo = rng.normal(0, 0.02, FS_EMG)

    return intensidade * muscular + fundo


# ======================================
# PROCESSAMENTO
# ======================================

def calcular_rms(sinal):
    return float(np.sqrt(np.mean(sinal ** 2)))


def calcular_mu(eeg):
    freq = np.fft.rfftfreq(
        len(eeg), 1 / FS_EEG
    )

    espectro = np.abs(np.fft.rfft(eeg)) ** 2

    banda = (freq >= 8) & (freq <= 13)

    return float(
        np.sum(espectro[banda]) / len(eeg) ** 2
    )


def normalizar_ativacao(rms):
    return float(
        np.clip((rms - 0.05) / 0.95, 0, 1)
    )


# ======================================
# CLASSIFICACAO EEG + sEMG
# ======================================

def classificar(eeg, flexor, extensor):
    mu = calcular_mu(eeg)

    af = normalizar_ativacao(
        calcular_rms(flexor)
    )

    ae = normalizar_ativacao(
        calcular_rms(extensor)
    )

    # EEG indica repouso
    if mu >= 30:
        return 0, 0.0, mu, af, ae

    # Flexor dominante
    if af - ae > 0.15:
        return 1, af, mu, af, ae

    # Extensor dominante
    if ae - af > 0.15:
        return 2, ae, mu, af, ae

    return 0, 0.0, mu, af, ae


# ======================================
# COMUNICACAO COM COPPELIASIM
# ======================================

def enviar_comando(comando, intensidade):
    sim.setFloatProperty(
        sim.handle_scene,
        "signal.inmoov_intensidade",
        float(intensidade)
    )

    sim.setIntProperty(
        sim.handle_scene,
        "signal.inmoov_comando",
        int(comando)
    )


def ler_indicador():
    return {
        nome: float(np.degrees(
            sim.getJointPosition(junta)
        ))
        for nome, junta in juntas.items()
    }


# ======================================
# EXPERIMENTO
# ======================================

def obter_cenario(tempo):
    if tempo < 4:
        return "FECHAR", 1, 0.8, 0.02

    if tempo < 7:
        return "REPOUSO", 0, 0.02, 0.02

    if tempo < 11:
        return "ABRIR", 1, 0.02, 0.8

    if tempo < 14:
        return "REPOUSO", 0, 0.02, 0.02

    if tempo < 18:
        return "FECHAR", 1, 0.8, 0.02

    if tempo < 21:
        return "REPOUSO", 0, 0.02, 0.02

    return "ABRIR", 1, 0.02, 0.8


def executar():
    print("Aguardando Play no CoppeliaSim...")

    while sim.getSimulationState() == sim.simulation_stopped:
        time.sleep(0.2)

    print("\nExperimento iniciado!")

    inicio = time.monotonic()

    try:
        while True:
            tempo = time.monotonic() - inicio

            if tempo >= DURACAO:
                break

            if sim.getSimulationState() == sim.simulation_stopped:
                print("Simulacao interrompida.")
                break

            # Definir condicao experimental
            cenario, atividade, nf, ne = obter_cenario(tempo)

            # Gerar sinais biologicos sinteticos
            eeg = gerar_eeg(atividade, tempo)
            flexor = gerar_semg(nf)
            extensor = gerar_semg(ne)

            # Processar e classificar
            comando, intensidade, mu, af, ae = classificar(
                eeg, flexor, extensor
            )

            # Enviar comando classificado
            enviar_comando(comando, intensidade)

            # Medir resposta do robo
            angulos = ler_indicador()

            # Registrar janelas dos sinais
            inicio_janela = max(0, tempo - INTERVALO)

            n_eeg = max(1, int(FS_EEG * INTERVALO))
            n_emg = max(1, int(FS_EMG * INTERVALO))

            te = np.linspace(
                inicio_janela, tempo, n_eeg
            )

            tm = np.linspace(
                inicio_janela, tempo, n_emg
            )

            tempos_eeg.extend(te)
            valores_eeg.extend(eeg[-n_eeg:])

            tempos_emg.extend(tm)
            valores_flexor.extend(flexor[-n_emg:])
            valores_extensor.extend(extensor[-n_emg:])

            tempos_controle.append(tempo)
            comandos.append(comando)
            angulos_indicador.append(angulos["base"])

            potencias_mu.append(mu)
            rms_flexor.append(calcular_rms(flexor))
            rms_extensor.append(calcular_rms(extensor))

            nomes = {
                0: "REPOUSO",
                1: "FECHAR",
                2: "ABRIR"
            }

            print(
                f"{tempo:5.1f}s | "
                f"EEG mu: {mu:5.2f} | "
                f"Flexor: {af:.2f} | "
                f"Extensor: {ae:.2f} | "
                f"Comando: {nomes[comando]:7s} | "
                f"Indicador: {angulos['base']:5.1f} graus"
            )

            time.sleep(INTERVALO)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario.")

    finally:
        try:
            if sim.getSimulationState() != sim.simulation_stopped:
                enviar_comando(0, 0.0)
        except Exception as erro:
            print(f"Falha ao enviar repouso: {erro}")

        print("\nExperimento encerrado.")


# ======================================
# GRAFICOS
# ======================================

def gerar_graficos():
    if not tempos_controle:
        print("Nenhum dado registrado.")
        return

    fig, axes = plt.subplots(
        5, 1, figsize=(13, 12)
    )

    axes[0].plot(tempos_eeg, valores_eeg, linewidth=0.6)
    axes[0].set_title("EEG sintetico")
    axes[0].set_ylabel("Amplitude (uV)")

    axes[1].plot(
        tempos_emg, valores_flexor,
        linewidth=0.5
    )
    axes[1].set_title("sEMG flexor")
    axes[1].set_ylabel("Amplitude (u.a.)")

    axes[2].plot(
        tempos_emg, valores_extensor,
        linewidth=0.5
    )
    axes[2].set_title("sEMG extensor")
    axes[2].set_ylabel("Amplitude (u.a.)")

    axes[3].step(
        tempos_controle, comandos,
        where="post"
    )
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(
        ["REPOUSO", "FECHAR", "ABRIR"]
    )
    axes[3].set_title("Comandos classificados")

    axes[4].plot(
        tempos_controle,
        angulos_indicador,
        marker=".",
        markersize=2
    )
    axes[4].set_title(
        "Resposta real da articulacao index_base"
    )
    axes[4].set_ylabel("Angulo (graus)")
    axes[4].set_xlabel("Tempo (s)")

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()

    plt.savefig(
        "experimento_eeg_semg_inmoov.png",
        dpi=150
    )

    plt.close(fig)

    print("Graficos salvos com sucesso!")


if __name__ == "__main__":
    executar()
    gerar_graficos()
