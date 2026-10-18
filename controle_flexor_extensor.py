
import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Conexão com o CoppeliaSim
client = RemoteAPIClient()
sim = client.require("sim")

# Articulações dos dedos
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

rng = np.random.default_rng(42)

# Parâmetros do controlador
ANGULO_MAXIMO = 30.0
LIMIAR = 0.05
RMS_MAXIMO = 1.0

PASSO = 0.1
VELOCIDADE_MAXIMA = 10.0  # Graus por segundo
ALPHA = 0.15              # Suavização do sinal
LIMIAR_MOVIMENTO = 0.15

posicao_atual = 0.0

# Valores utilizados pelo filtro
flexor_suavizado = 0.0
extensor_suavizado = 0.0


def gerar_semg(intensidade, amostras=200):
    """Gera um sinal muscular sintético com ruído."""

    ruido_muscular = rng.normal(0, 1, amostras)
    ruido_base = rng.normal(0, 0.02, amostras)

    return intensidade * ruido_muscular + ruido_base


def calcular_ativacao(sinal):
    """Calcula o RMS e normaliza a ativação muscular."""

    rms = float(np.sqrt(np.mean(sinal ** 2)))

    ativacao = np.clip(
        (rms - LIMIAR) / (RMS_MAXIMO - LIMIAR),
        0,
        1
    )

    return float(ativacao)


def suavizar_sinal(valor_atual, valor_anterior):
    """Filtro exponencial para reduzir oscilações."""

    return (
        ALPHA * valor_atual
        + (1 - ALPHA) * valor_anterior
    )


def classificar_movimento(flexor, extensor):
    """Identifica flexão, extensão ou repouso."""

    diferenca = flexor - extensor

    if diferenca > LIMIAR_MOVIMENTO:
        return "FECHAR"

    elif diferenca < -LIMIAR_MOVIMENTO:
        return "ABRIR"

    return "MANTER"


def mover_mao(angulo):
    """Envia a posição desejada às articulações."""

    angulo = float(
        np.clip(angulo, 0, ANGULO_MAXIMO)
    )

    for junta in juntas.values():
        sim.setJointTargetPosition(
            junta,
            math.radians(angulo)
        )


def executar_estado(nome, flexor, extensor, duracao=3):
    global posicao_atual
    global flexor_suavizado
    global extensor_suavizado

    print(f"\nEstado: {nome}")

    inicio = time.monotonic()
    ultimo_passo = inicio

    while time.monotonic() - inicio < duracao:

        agora = time.monotonic()
        dt = min(agora - ultimo_passo, 0.2)
        ultimo_passo = agora

        # Gerar sinais sEMG independentes
        sinal_flexor = gerar_semg(flexor)
        sinal_extensor = gerar_semg(extensor)

        # Calcular ativação muscular
        ativacao_flexor = calcular_ativacao(sinal_flexor)
        ativacao_extensor = calcular_ativacao(sinal_extensor)

        # Suavizar as ativações
        flexor_suavizado = suavizar_sinal(
            ativacao_flexor,
            flexor_suavizado
        )

        extensor_suavizado = suavizar_sinal(
            ativacao_extensor,
            extensor_suavizado
        )

        # Classificar movimento
        comando = classificar_movimento(
            flexor_suavizado,
            extensor_suavizado
        )

        # Diferença de atividade muscular
        diferenca = (
            flexor_suavizado - extensor_suavizado
        )

        # Controlar velocidade dos dedos
        if comando == "FECHAR":
            variacao = (
                VELOCIDADE_MAXIMA
                * diferenca
                * dt
            )

            posicao_atual += variacao

        elif comando == "ABRIR":
            variacao = (
                VELOCIDADE_MAXIMA
                * diferenca
                * dt
            )

            posicao_atual += variacao

        # Limitar a posição
        posicao_atual = float(
            np.clip(
                posicao_atual,
                0,
                ANGULO_MAXIMO
            )
        )

        # Enviar movimento ao CoppeliaSim
        mover_mao(posicao_atual)

        print(
            f"Flexor: {flexor_suavizado:.2f} | "
            f"Extensor: {extensor_suavizado:.2f} | "
            f"Comando: {comando} | "
            f"Ângulo: {posicao_atual:.1f}°"
        )

        time.sleep(PASSO)


def main():
    global posicao_atual
    global flexor_suavizado
    global extensor_suavizado

    posicao_atual = 0.0
    flexor_suavizado = 0.0
    extensor_suavizado = 0.0

    try:
        sim.startSimulation()
        time.sleep(1)

        mover_mao(0)
        time.sleep(1)

        executar_estado("Repouso inicial", 0.02, 0.02, 3)

        executar_estado("Flexão", 1.0, 0.02, 5)

        executar_estado("Repouso intermediário", 0.02, 0.02, 3)

        executar_estado("Extensão", 0.02, 1.0, 5)

        executar_estado("Repouso final", 0.02, 0.02, 3)

    finally:
        sim.stopSimulation()
        print("\nSimulação encerrada!")


if __name__ == "__main__":
    main()
