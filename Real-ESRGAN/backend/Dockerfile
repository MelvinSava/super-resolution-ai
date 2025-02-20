# Utiliser une image Python optimisée
FROM python:3.12-slim

# Définir le dossier de travail
WORKDIR /app

# Copier uniquement le backend
COPY backend/ /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Flask
EXPOSE 5000

# Lancer Flask
CMD ["python", "app.py"]
