
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# CONEXÃO COM O COPPELIASIM

client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# ARTICULAÇÕES DO INDICADOR

articulacoes = {
    "base": sim.getObject("/index_base"),
    "meio": sim.getObject("/index_middle"),
    "ponta": sim.getObject("/index_tip")
}

# CONFIGURAÇÕES DO MOVIMENTO

AMPLITUDE_TESTE = 10.0
DURACAO_MOVIMENTO = 3.0
INTERVALO = 0.05

# DEFINIR ÂNGULOS

def definir_angulos(angulos):
    """
    Envia comandos apenas para as
    articulações selecionadas.
    """

    for nome, angulo in angulos.items():

        junta = articulacoes[nome]

        sim.setJointTargetPosition(
            junta,
            math.radians(float(angulo))
        )

# MOVIMENTAÇÃO SUAVE

def movimentar(inicio, destino, duracao):
    """
    Movimenta as articulações selecionadas
    utilizando interpolação suave.
    """

    passos = max(1, int(duracao / INTERVALO))

    for passo in range(passos + 1):

        progresso = passo / passos

        # Interpolação Smoothstep
        suave = progresso**2 * (3 - 2 * progresso)

        angulos = {}

        # Percorrer somente as articulações
        # presentes neste teste
        for nome in inicio:

            angulo_inicial = inicio[nome]
            angulo_final = destino[nome]

            angulos[nome] = (
                angulo_inicial
                + (angulo_final - angulo_inicial) * suave
            )

        definir_angulos(angulos)

        time.sleep(INTERVALO)


#  TESTAR UMA ARTICULAÇÃO
def testar_articulacao(nome):
    """
    Testa uma articulação do indicador,
    fazendo-a avançar e retornar.
    """

    print(f"\nTestando articulação: {nome}")

    inicio = {nome: 0.0}

    destino = {nome: AMPLITUDE_TESTE}

    print("Movimentando para 10 graus...")

    movimentar(
        inicio,
        destino,
        DURACAO_MOVIMENTO
    )

    time.sleep(1)

    print("Retornando para 0 graus...")

    movimentar(
        destino,
        inicio,
        DURACAO_MOVIMENTO
    )

    time.sleep(1)

    print(f"Teste da articulação {nome} concluído!")

# EXECUÇÃO PRINCIPAL

def main():

    try:

        print("\nIniciando simulação...")

        sim.startSimulation()

        time.sleep(1)

        # Testar articulação da base
        testar_articulacao("base")

        # Testar articulação do meio
        testar_articulacao("meio")

        # Testar articulação da ponta
        testar_articulacao("ponta")

        print("\nTodos os testes foram concluídos!")

    finally:

        sim.stopSimulation()

        print("\nSimulação encerrada!")

# INICIAR PROGRAMA


if __name__ == "__main__":
    main()
