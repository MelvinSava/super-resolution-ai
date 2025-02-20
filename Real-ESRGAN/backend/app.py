import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image
from RealESRGAN import RealESRGAN
import torch
import logging
import time


torch.set_num_threads(8)  
torch.backends.mkldnn.enabled = True  
torch.backends.openmp.enabled = True  


# Configuration du logger
logging.basicConfig(level=logging.DEBUG)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Permet toutes les origines

# Dossiers pour stocker les fichiers téléchargés et les résultats
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Charger le modèle RealESRGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4plus.pth', download=True)

@app.route("/")
def home():
    return render_template('index.html')



@app.route("/upload", methods=["POST"])
def upload_file():
    app.logger.debug("Requête POST reçue")

    if "file" not in request.files:
        app.logger.error("Aucun fichier trouvé dans la requête")
        return jsonify({"error": "Aucun fichier reçu"}), 400

    file = request.files["file"]
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    try:
        start_time = time.time()  # Début du traitement
        app.logger.debug(f"Traitement de l'image: {filename}")

        image = Image.open(filename).convert("RGB")
        sr_image = model.predict(image)

        output_filename = "sr_" + file.filename
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        sr_image.save(output_path)

        processing_time = time.time() - start_time  # Durée réelle du traitement
        app.logger.debug(f"Image améliorée sauvegardée: {output_path} en {processing_time:.2f} sec")

        return jsonify({
            "image_url": f"/results/{output_filename}",
            "processing_time": processing_time  # Retour du temps d'exécution
        })

    except Exception as e:
        app.logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
        return jsonify({"error": f"Erreur lors du traitement de l'image: {str(e)}"}), 500


@app.route('/results/<filename>')
def serve_image(filename):
    # Servir l'image améliorée depuis le répertoire 'results'
    try:
        app.logger.debug(f"Servir l'image: {filename}")
        return send_from_directory(RESULT_FOLDER, filename)
    except FileNotFoundError:
        app.logger.error(f"Image non trouvée: {filename}")
        return jsonify({"error": "Image non trouvée"}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
