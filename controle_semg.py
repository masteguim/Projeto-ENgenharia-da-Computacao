
import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Conexão com o CoppeliaSim
client = RemoteAPIClient()
sim = client.require("sim")

# Articulações da mão InMoov
nomes = [
    "index_base", "index_middle", "index_tip",
    "middle_base", "middle_middle", "middle_tip",
    "ring_base", "ring_middle", "ring_tip",
    "pinky_base", "pinky_middle", "pinky_tip",
    "thumb_base", "thumb_tip"
]

juntas = {
    nome: sim.getObject("/" + nome)
    for nome in nomes
}

# Gerador de sinais artificiais
rng = np.random.default_rng(42)


def gerar_semg(intensidade, amostras=200):
    """Gera um sinal sEMG sintético com ruído de fundo."""

    ruido_muscular = rng.normal(0, 1, amostras)
    ruido_base = rng.normal(0, 0.02, amostras)

    return intensidade * ruido_muscular + ruido_base


def calcular_rms(sinal):
    """Calcula a amplitude RMS do sinal sEMG."""

    return float(np.sqrt(np.mean(sinal ** 2)))


def calcular_ativacao(rms):
    """Converte o RMS em ativação muscular entre 0 e 1."""

    limiar = 0.05
    rms_maximo = 1.0

    if rms <= limiar:
        return 0.0

    ativacao = (rms - limiar) / (rms_maximo - limiar)

    return float(np.clip(ativacao, 0, 1))


def mover_mao(ativacao):
    """Converte a ativação muscular em posição das articulações."""

    angulo = 30 * np.clip(ativacao, 0, 1)

    for junta in juntas.values():
        sim.setJointTargetPosition(
            junta,
            math.radians(float(angulo))
        )


def testar_contracao(intensidade, duracao=3):
    """Simula uma contração muscular durante um intervalo."""

    inicio = time.monotonic()

    while time.monotonic() - inicio < duracao:

        # Gerar sinal muscular
        sinal = gerar_semg(intensidade)

        # Calcular RMS
        rms = calcular_rms(sinal)

        # Calcular ativação muscular
        ativacao = calcular_ativacao(rms)

        # Enviar comando ao CoppeliaSim
        mover_mao(ativacao)

        print(
            f"sEMG RMS: {rms:.3f} | "
            f"Ativação: {ativacao:.2f}"
        )

        time.sleep(0.1)


def main():
    try:
        # Iniciar simulação
        sim.startSimulation()
        time.sleep(1)

        print("\nRepouso muscular")
        testar_contracao(0.0)

        print("\nContração moderada")
        testar_contracao(0.5)

        print("\nContração forte")
        testar_contracao(1.0)

        print("\nRelaxamento")
        testar_contracao(0.0)

    finally:
        sim.stopSimulation()
        print("\nSimulação finalizada!")


if __name__ == "__main__":
    main()
