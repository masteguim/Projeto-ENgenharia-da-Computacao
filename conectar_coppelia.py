from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Conectar ao CoppeliaSim
client = RemoteAPIClient()
sim = client.require("sim")

print("Conectado ao CoppeliaSim!")

# Listar todas as articulações da cena
juntas = sim.getObjectsInTree(
    sim.handle_scene,
    sim.object_joint_type,
    0
)

print(f"Total de articulações: {len(juntas)}")

for junta in juntas:
    nome = sim.getObjectAlias(junta)
    print(f"Articulação: {nome} | Handle: {junta}")
