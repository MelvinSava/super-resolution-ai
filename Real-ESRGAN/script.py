from RealESRGAN import RealESRGAN
import torch
import os
from PIL import Image

# Choisir le périphérique (GPU si disponible, sinon CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Créer une instance du modèle RealESRGAN avec un facteur de mise à l'échelle de 4
model = RealESRGAN(device, scale=4)

# Charger les poids du modèle
weights_path = 'weights/RealESRGAN_x4plus.pth'
model.load_weights(weights_path, download=True)  # Télécharge si nécessaire

# Charger l'image à améliorer
path_to_image = 'Real-ESRGAN/inputs/00003.png'  # Chemin de l'image basse résolution

if not os.path.exists(path_to_image):
    raise FileNotFoundError(f"L'image '{path_to_image}' est introuvable. Vérifie le chemin.")

image = Image.open(path_to_image).convert('RGB')

# Prédire l'image haute résolution
sr_image = model.predict(image)

# Créer le dossier de sortie s'il n'existe pas
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder l'image restaurée
output_path = os.path.join(output_dir, "sr_image.png")
sr_image.save(output_path)

print(f"Image enregistrée : {output_path}")
