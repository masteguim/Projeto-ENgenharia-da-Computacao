
import time
import math
import csv
from datetime import datetime
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# 1. CONEXÃO COM O COPPELIASIM
# ==========================================

client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# ==========================================
# 2. ARTICULAÇÕES DA MÃO INMOOV
# ==========================================

nomes = [
    "wrist",
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

# ==========================================
# 3. CONFIGURAÇÕES DO MONITORAMENTO
# ==========================================

INTERVALO = 0.1
DURACAO = 30

# ==========================================
# 4. LEITURA DAS ARTICULAÇÕES
# ==========================================

def ler_articulacoes():
    """
    Lê os ângulos atuais das articulações.
    Não envia comandos aos motores.
    """

    angulos = {}

    for nome, junta in juntas.items():

        posicao = sim.getJointPosition(junta)

        angulos[nome] = math.degrees(posicao)

    return angulos


# ==========================================
# 5. MONITORAMENTO EM TEMPO REAL
# ==========================================

def monitorar():

    arquivo = "movimentos_inmoov.csv"

    print("\nIniciando monitoramento...")
    print("Nenhum comando será enviado aos motores.")

    inicio = time.monotonic()

    with open(arquivo, "w", newline="") as csvfile:

        escritor = csv.writer(csvfile)

        # Cabeçalho do arquivo CSV
        escritor.writerow(
            ["tempo"] + nomes
        )

        while time.monotonic() - inicio < DURACAO:

            tempo = time.monotonic() - inicio

            angulos = ler_articulacoes()

            escritor.writerow(
                [round(tempo, 3)]
                + [round(angulos[nome], 3) for nome in nomes]
            )

            print(f"\nTempo: {tempo:.1f}s")

            for nome, angulo in angulos.items():

                print(
                    f"{nome}: {angulo:.2f}°"
                )

            time.sleep(INTERVALO)

    print("\nMonitoramento concluído!")
    print(f"Dados salvos em: {arquivo}")


# ==========================================
# 6. EXECUÇÃO PRINCIPAL
# ==========================================

def main():

    # Verifica se a simulação está rodando
    if sim.getSimulationState() == sim.simulation_stopped:

        print("Aguardando a simulação iniciar...")

        while sim.getSimulationState() == sim.simulation_stopped:
            time.sleep(0.5)

    monitorar()


if __name__ == "__main__":
    main()
