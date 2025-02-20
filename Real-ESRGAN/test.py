import torch
print("PyTorch détecte CUDA :", torch.cuda.is_available())
print("Nombre de GPU :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Nom du GPU utilisé :", torch.cuda.get_device_name(0))
