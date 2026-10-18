
import time
import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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

print("Articulações encontradas!")

def mover_mao(angulo):
    for nome, junta in juntas.items():
        sim.setJointTargetPosition(
            junta,
            math.radians(angulo)
        )

# Iniciar simulação
sim.startSimulation()
time.sleep(1)

try:
    print("Movendo as articulações para 0 graus...")
    mover_mao(0)
    time.sleep(3)

    print("Movendo as articulações para 30 graus...")
    mover_mao(30)
    time.sleep(3)

    print("Retornando para 0 graus...")
    mover_mao(0)
    time.sleep(3)

finally:
    sim.stopSimulation()
    print("Simulação encerrada.")
