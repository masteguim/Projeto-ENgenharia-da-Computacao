
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =====================================
# CONFIGURAÇÕES
# =====================================

FS_EEG = 256
FS_EMG = 1000
INTERVALO = 0.2
DURACAO = 30

rng = np.random.default_rng(42)

# Conexão com CoppeliaSim
client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# Filtro muscular de 20 a 400 Hz
filtro_emg = butter(
    4,
    [20, 400],
    btype="bandpass",
    fs=FS_EMG,
    output="sos"
)

# Histórico para os gráficos
historico_tempo = []
historico_eeg = []
historico_flexor = []
historico_extensor = []
historico_comandos = []

# =====================================
# GERAR EEG SINTÉTICO
# =====================================

def gerar_eeg(atividade):
    t = np.arange(FS_EEG) / FS_EEG

    # Redução da atividade mu
    # durante intenção motora simulada
    amplitude_mu = 15 if atividade == 0 else 6

    sinal = (
        amplitude_mu * np.sin(2 * np.pi * 10 * t)
        + 5 * np.sin(2 * np.pi * 20 * t)
        + rng.normal(0, 3, FS_EEG)
    )

    return sinal


# =====================================
# GERAR sEMG SINTÉTICO
# =====================================

def gerar_semg(intensidade):
    ruido = rng.normal(0, 1, FS_EMG)

    muscular = sosfiltfilt(
        filtro_emg, ruido
    )

    # Normalizar RMS da componente muscular
    rms = np.sqrt(np.mean(muscular ** 2))
    muscular = muscular / max(rms, 1e-8)

    # Ruído de fundo
    fundo = rng.normal(0, 0.02, FS_EMG)

    return intensidade * muscular + fundo


# =====================================
# PROCESSAMENTO
# =====================================

def potencia_mu(eeg):
    frequencias = np.fft.rfftfreq(
        len(eeg), 1 / FS_EEG
    )

    espectro = np.abs(np.fft.rfft(eeg)) ** 2

    banda = (
        (frequencias >= 8)
        & (frequencias <= 13)
    )

    return float(
        np.sum(espectro[banda]) / len(eeg) ** 2
    )


def calcular_rms(sinal):
    return float(
        np.sqrt(np.mean(sinal ** 2))
    )


def ativacao_muscular(rms):
    return float(
        np.clip((rms - 0.05) / 0.95, 0, 1)
    )


# =====================================
# CLASSIFICADOR EEG + sEMG
# =====================================

def classificar(eeg, flexor, extensor):
    mu = potencia_mu(eeg)

    af = ativacao_muscular(calcular_rms(flexor))
    ae = ativacao_muscular(calcular_rms(extensor))

    # Repouso cerebral simulado
    if mu >= 30:
        return 0, 0.0, mu, af, ae

    # Flexor dominante: fechar
    if af - ae > 0.15:
        return 1, af, mu, af, ae

    # Extensor dominante: abrir
    if ae - af > 0.15:
        return 2, ae, mu, af, ae

    return 0, 0.0, mu, af, ae


# =====================================
# ENVIO AO COPPELIASIM
# =====================================

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


# =====================================
# SALVAR GRÁFICOS
# =====================================

def salvar_graficos():
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9)
    )

    t = np.array(historico_tempo)

    axes[0].plot(t, historico_eeg)
    axes[0].set_title("Potência mu do EEG sintético")
    axes[0].set_ylabel("Potência")

    axes[1].plot(t, historico_flexor)
    axes[1].set_title("sEMG flexor - RMS")
    axes[1].set_ylabel("RMS")

    axes[2].plot(t, historico_extensor)
    axes[2].set_title("sEMG extensor - RMS")
    axes[2].set_ylabel("RMS")

    axes[3].step(
        t, historico_comandos, where="post"
    )
    axes[3].set_title("Comandos classificados")
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(
        ["REPOUSO", "FECHAR", "ABRIR"]
    )
    axes[3].set_xlabel("Tempo (s)")

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("resultado_eeg_semg.png", dpi=150)
    plt.close(fig)


# =====================================
# EXECUÇÃO PRINCIPAL
# =====================================

def main():
    nomes = {
        0: "REPOUSO",
        1: "FECHAR",
        2: "ABRIR"
    }

    print("Aguardando Play no CoppeliaSim...")

    while sim.getSimulationState() == sim.simulation_stopped:
        time.sleep(0.2)

    print("Simulação EEG + sEMG iniciada!")

    inicio = time.monotonic()

    try:
        while time.monotonic() - inicio < DURACAO:
            if sim.getSimulationState() == sim.simulation_stopped:
                print("CoppeliaSim foi interrompido.")
                break

            t = time.monotonic() - inicio

            # Cenários experimentais
            fase = int(t // 4) % 3

            if fase == 0:
                atividade = 1
                nivel_flexor = 0.8
                nivel_extensor = 0.02

            elif fase == 1:
                atividade = 1
                nivel_flexor = 0.02
                nivel_extensor = 0.8

            else:
                atividade = 0
                nivel_flexor = 0.02
                nivel_extensor = 0.02

            # Gerar sinais
            eeg = gerar_eeg(atividade)
            flexor = gerar_semg(nivel_flexor)
            extensor = gerar_semg(nivel_extensor)

            # Classificar a partir dos sinais
            comando, intensidade, mu, af, ae = classificar(
                eeg, flexor, extensor
            )

            # Enviar comando ao InMoov
            enviar_comando(comando, intensidade)

            # Registrar resultados
            historico_tempo.append(t)
            historico_eeg.append(mu)
            historico_flexor.append(calcular_rms(flexor))
            historico_extensor.append(calcular_rms(extensor))
            historico_comandos.append(comando)

            print(
                f"{t:5.1f}s | "
                f"EEG mu: {mu:6.2f} | "
                f"Flexor: {af:.2f} | "
                f"Extensor: {ae:.2f} | "
                f"{nomes[comando]} | "
                f"Intensidade: {intensidade:.2f}"
            )

            time.sleep(INTERVALO)

    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")

    finally:
        try:
            if sim.getSimulationState() != sim.simulation_stopped:
                enviar_comando(0, 0.0)
        except Exception as erro:
            print(f"Não foi possível enviar repouso: {erro}")

        if historico_tempo:
            salvar_graficos()
            print("Gráficos salvos em resultado_eeg_semg.png")

        print("Experimento finalizado.")


if __name__ == "__main__":
    main()
