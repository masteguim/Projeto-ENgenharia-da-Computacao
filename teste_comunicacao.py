
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado!")

def enviar(comando, intensidade):
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

try:
    print("Aguardando Play...")

    while sim.getSimulationState() == sim.simulation_stopped:
        time.sleep(0.2)

    print("FECHAR")
    enviar(1, 0.8)
    time.sleep(4)

    print("REPOUSO")
    enviar(0, 0)
    time.sleep(2)

    print("ABRIR")
    enviar(2, 1.0)
    time.sleep(4)

    print("REPOUSO")
    enviar(0, 0)

finally:
    try:
        if sim.getSimulationState() != sim.simulation_stopped:
            enviar(0, 0)
    except Exception:
        pass

    print("Teste finalizado.")
