
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# CONFIGURACOES

FS_EEG = 256
FS_EMG = 1000

INTERVALO = 0.15
DURACAO = 30

rng = np.random.default_rng(42)

DEDOS = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky"
]

NOMES = [
    "Polegar",
    "Indicador",
    "Medio",
    "Anelar",
    "Mindinho"
]

COMANDOS = {
    0: "REPOUSO",
    1: "FECHAR",
    2: "ABRIR"
}

# CONEXAO COM COPPELIASIM

print("Conectando ao CoppeliaSim...")

client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# IDENTIFICAR ARTICULACOES

juntas = {}

for nome in DEDOS:
    juntas[nome] = sim.getObject(
        "/" + nome + "_base"
    )

print("Articulacoes identificadas!")

# FILTRO sEMG

filtro_emg = butter(
    4,
    [20, 400],
    btype="bandpass",
    fs=FS_EMG,
    output="sos"
)

# HISTORICO DO EXPERIMENTO

historico = {
    "tempo": [],
    "mu": [],
    "flexor": [],
    "extensor": [],
    "comando": [],
    "dedo": [],
    "intensidade": [],
    "angulos": {
        nome: [] for nome in DEDOS
    }
}

# GERACAO DE EEG SINTETICO

def gerar_eeg(atividade, tempo):

    t = tempo + np.arange(FS_EEG) / FS_EEG

    # Ritmo mu de 10 Hz.
    # A amplitude diminui durante
    # a atividade motora simulada.

    amplitude_mu = 6 if atividade else 15

    ritmo_mu = (
        amplitude_mu
        * np.sin(2 * np.pi * 10 * t)
    )

    ritmo_beta = (
        5 * np.sin(2 * np.pi * 20 * t)
    )

    ruido = rng.normal(
        0, 3, FS_EEG
    )

    return ritmo_mu + ritmo_beta + ruido

# GERACAO DE sEMG SINTETICO

def gerar_semg(intensidade):

    ruido = rng.normal(
        0, 1, FS_EMG
    )

    # Gerar componente muscular filtrada
    muscular = sosfiltfilt(
        filtro_emg,
        ruido
    )

    # Normalizar a energia do sinal
    energia = np.sqrt(
        np.mean(muscular ** 2)
    )

    muscular = muscular / max(
        energia, 1e-8
    )

    # Adicionar ruido de fundo
    fundo = rng.normal(
        0, 0.02, FS_EMG
    )

    return (
        intensidade * muscular
        + fundo
    )

# PROCESSAMENTO DOS SINAIS

def calcular_rms(sinal):

    return float(
        np.sqrt(
            np.mean(sinal ** 2)
        )
    )


def calcular_potencia_mu(eeg):

    frequencias = np.fft.rfftfreq(
        len(eeg),
        d=1 / FS_EEG
    )

    espectro = (
        np.abs(np.fft.rfft(eeg)) ** 2
    )

    banda_mu = (
        (frequencias >= 8)
        & (frequencias <= 13)
    )

    return float(
        np.sum(espectro[banda_mu])
        / len(eeg) ** 2
    )


def calcular_ativacao(sinal):

    valor_rms = calcular_rms(sinal)

    ativacao = (
        (valor_rms - 0.05) / 0.95
    )

    return float(
        np.clip(ativacao, 0, 1)
    )

# CLASSIFICADOR EEG + sEMG

def classificar(eeg, flexor, extensor):

    mu = calcular_potencia_mu(eeg)

    ativacao_flexor = calcular_ativacao(
        flexor
    )

    ativacao_extensor = calcular_ativacao(
        extensor
    )

    # EEG indica repouso
    if mu >= 30:
        return (
            0,
            0.0,
            mu,
            ativacao_flexor,
            ativacao_extensor
        )

    # Atividade muscular flexora
    diferenca = (
        ativacao_flexor
        - ativacao_extensor
    )

    if diferenca > 0.15:
        return (
            1,
            ativacao_flexor,
            mu,
            ativacao_flexor,
            ativacao_extensor
        )

    # Atividade muscular extensora
    if diferenca < -0.15:
        return (
            2,
            ativacao_extensor,
            mu,
            ativacao_flexor,
            ativacao_extensor
        )

    # Nenhum movimento detectado
    return (
        0,
        0.0,
        mu,
        ativacao_flexor,
        ativacao_extensor
    )


# ENVIAR COMANDOS AO INMOOV

def enviar_comando(comando, intensidade, dedo):

    # Selecionar o dedo
    sim.setIntProperty(
        sim.handle_scene,
        "signal.inmoov_dedo",
        int(dedo)
    )

    # Intensidade muscular
    sim.setFloatProperty(
        sim.handle_scene,
        "signal.inmoov_intensidade",
        float(intensidade)
    )

    # Movimento
    sim.setIntProperty(
        sim.handle_scene,
        "signal.inmoov_comando",
        int(comando)
    )


# SEQUENCIA EXPERIMENTAL

def obter_cenario(tempo):

    # Cada dedo recebe 6 segundos
    indice = min(
        int(tempo // 6),
        len(DEDOS) - 1
    )

    fase = tempo % 6

    # 2 segundos: flexao
    if fase < 2:
        return (
            indice,
            "FECHAR",
            1,
            0.9,
            0.02
        )

    # 1 segundo: repouso
    if fase < 3:
        return (
            indice,
            "REPOUSO",
            0,
            0.02,
            0.02
        )

    # 2 segundos: extensao
    if fase < 5:
        return (
            indice,
            "ABRIR",
            1,
            0.02,
            0.9
        )

    # 1 segundo: repouso
    return (
        indice,
        "REPOUSO",
        0,
        0.02,
        0.02
    )


# REGISTRAR MOVIMENTOS

def registrar(
    tempo,
    mu,
    flexor,
    extensor,
    comando,
    dedo,
    intensidade
):

    h = historico

    h["tempo"].append(tempo)
    h["mu"].append(mu)
    h["flexor"].append(flexor)
    h["extensor"].append(extensor)
    h["comando"].append(comando)
    h["dedo"].append(dedo)
    h["intensidade"].append(intensidade)

    for nome, junta in juntas.items():

        posicao = sim.getJointPosition(
            junta
        )

        angulo = np.degrees(
            posicao
        )

        h["angulos"][nome].append(
            float(angulo)
        )

# GERAR GRAFICOS

def salvar_graficos():

    h = historico

    if not h["tempo"]:
        print("Nenhum dado registrado.")
        return

    tempo = h["tempo"]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13, 13),
        sharex=True
    )

    # EEG
    axes[0].plot(
        tempo,
        h["mu"],
        label="Potencia mu"
    )

    axes[0].set_title(
        "EEG sintetico - Potencia da banda mu"
    )

    axes[0].set_ylabel("Potencia")

    # sEMG
    axes[1].plot(
        tempo,
        h["flexor"],
        label="Flexor"
    )

    axes[1].plot(
        tempo,
        h["extensor"],
        label="Extensor"
    )

    axes[1].set_title(
        "sEMG - Ativacao muscular"
    )

    axes[1].set_ylabel("Ativacao")
    axes[1].legend()

    # Comandos
    axes[2].step(
        tempo,
        h["comando"],
        where="post"
    )

    axes[2].set_title(
        "Comandos classificados"
    )

    axes[2].set_yticks(
        [0, 1, 2]
    )

    axes[2].set_yticklabels(
        ["REPOUSO", "FECHAR", "ABRIR"]
    )

    # Dedo selecionado
    axes[3].step(
        tempo,
        h["dedo"],
        where="post"
    )

    axes[3].set_title(
        "Dedo selecionado"
    )

    axes[3].set_yticks(
        [1, 2, 3, 4, 5]
    )

    axes[3].set_yticklabels(
        NOMES
    )

    # Resposta angular
    for nome, nome_exibicao in zip(
        DEDOS,
        NOMES
    ):

        axes[4].plot(
            tempo,
            h["angulos"][nome],
            label=nome_exibicao
        )

    axes[4].set_title(
        "Resposta angular dos dedos InMoov"
    )

    axes[4].set_ylabel(
        "Angulo (graus)"
    )

    axes[4].set_xlabel(
        "Tempo (segundos)"
    )

    axes[4].legend()

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()

    plt.savefig(
        "resultado_dedos_eeg_semg.png",
        dpi=150
    )

    plt.close(fig)

    print(
        "Graficos salvos em "
        "resultado_dedos_eeg_semg.png"
    )


# AGUARDAR INICIO DO COPPELIASIM

def aguardar_simulacao():

    estado = sim.getSimulationState()

    # Caso a simulacao esteja ativa,
    # encerrar para comecar um novo teste.
    if estado != sim.simulation_stopped:

        print(
            "Simulacao anterior detectada."
        )

        print(
            "Solicitando parada..."
        )

        sim.stopSimulation()

        while (
            sim.getSimulationState()
            != sim.simulation_stopped
        ):
            time.sleep(0.2)

    print()
    print(
        "Aguardando voce apertar Play "
        "no CoppeliaSim..."
    )

    # Esperar o inicio manual
    while (
        sim.getSimulationState()
        == sim.simulation_stopped
    ):
        time.sleep(0.2)

    print()
    print("Simulacao iniciada!")
    print(
        "Iniciando sinais EEG e sEMG..."
    )


# EXPERIMENTO PRINCIPAL

def main():

    aguardar_simulacao()

    inicio = time.monotonic()

    ultimo_estado = None

    try:

        while True:

            tempo = (
                time.monotonic() - inicio
            )

            # Duracao total
            if tempo >= DURACAO:
                break

            # Verificar o CoppeliaSim
            if (
                sim.getSimulationState()
                == sim.simulation_stopped
            ):
                print(
                    "Simulacao interrompida."
                )
                break

            # Definir dedo e atividade
            (
                indice,
                cenario,
                atividade,
                nivel_flexor,
                nivel_extensor
            ) = obter_cenario(tempo)

            # Gerar EEG
            eeg = gerar_eeg(
                atividade,
                tempo
            )

            # Gerar sEMG
            flexor = gerar_semg(
                nivel_flexor
            )

            extensor = gerar_semg(
                nivel_extensor
            )

            # Classificar sinais
            (
                comando,
                intensidade,
                mu,
                af,
                ae
            ) = classificar(
                eeg,
                flexor,
                extensor
            )

            # Selecionar dedo
            dedo = indice + 1

            # Enviar comando externo
            enviar_comando(
                comando,
                intensidade,
                dedo
            )

            # Registrar resultados
            registrar(
                tempo,
                mu,
                af,
                ae,
                comando,
                dedo,
                intensidade
            )

            # Mostrar alteracao de estado
            estado = (dedo, cenario)

            if estado != ultimo_estado:

                print()
                print(
                    f"Dedo: {NOMES[indice]}"
                )

                print(
                    f"Estado experimental: "
                    f"{cenario}"
                )

                ultimo_estado = estado

            print(
                f"{tempo:5.1f}s | "
                f"Mu: {mu:6.2f} | "
                f"Flexor: {af:.2f} | "
                f"Extensor: {ae:.2f} | "
                f"Comando: {COMANDOS[comando]} | "
                f"Intensidade: {intensidade:.2f}"
            )

            time.sleep(INTERVALO)

    except KeyboardInterrupt:

        print()
        print(
            "Experimento interrompido."
        )

    finally:

        # Deixar controlador em repouso
        try:

            if (
                sim.getSimulationState()
                != sim.simulation_stopped
            ):

                enviar_comando(
                    0,
                    0,
                    1
                )

        except Exception as erro:

            print(
                "Erro ao enviar repouso:",
                erro
            )

        # Salvar resultado
        salvar_graficos()

        print()
        print(
            "Experimento finalizado."
        )

# INICIO DO PROGRAMA

if __name__ == "__main__":
    main()
