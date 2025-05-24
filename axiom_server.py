import os
import uvicorn
import requests
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
TEXT_MODEL_NAME = "llama3:8b"
VISION_MODEL_NAME = "llava" # Assurez-vous d'avoir fait 'ollama pull llava'

# --- Initialisation de l'application FastAPI ---
app = FastAPI()

# --- Middleware CORS ---
# Permet à notre page web (servie sur un domaine) de communiquer avec notre API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Accepte les requêtes de n'importe où (simple pour le dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Structure des données pour les requêtes ---
class ChatRequest(BaseModel):
    text: str
    image: Optional[str] = None # L'image sera une chaîne de caractères en Base64
    history: List[dict]

# --- Servir les fichiers statiques (CSS) ---
# Le dossier 'static' contiendra notre fichier style.css
# Créez un dossier nommé 'static' et mettez y style.css
if not os.path.exists('static'):
    os.makedirs('static')
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Route Principale pour l'interface web ---
@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Sert notre fichier d'interface index.html."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>Fichier index.html non trouvé.</h1>")

# --- Route API pour le chat ---
@app.post("/api/chat")
async def chat_with_axiom(request: ChatRequest):
    """
    Reçoit le message de l'utilisateur, interagit avec Ollama et renvoie la réponse.
    """
    conversation_history = request.history
    current_model = TEXT_MODEL_NAME
    
    # Prépare le nouveau message de l'utilisateur
    user_message = {"role": "user", "content": request.text}

    # Si une image est envoyée, on utilise le modèle de vision (LLaVA)
    if request.image:
        print("🖼️ Image détectée, utilisation du modèle de vision LLaVA.")
        current_model = VISION_MODEL_NAME
        # Le format pour llava avec image est un peu différent
        user_message["images"] = [request.image]
    
    conversation_history.append(user_message)

    # Prépare le payload pour Ollama
    payload = {
        "model": current_model,
        "messages": conversation_history,
        "stream": False
    }

    print(f"🤖 Envoi de la requête à Ollama avec le modèle {current_model}...")

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status() # Lève une exception si erreur HTTP
        
        response_json = response.json()
        assistant_message = response_json.get("message", {})
        
        # Ajoute la réponse de l'assistant à l'historique pour le prochain tour
        conversation_history.append(assistant_message)
        
        return {"response": assistant_message.get("content", ""), "history": conversation_history}

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de communication avec Ollama: {e}")
        return {"response": f"Désolé, une erreur est survenue en contactant le moteur Axiom. Détails: {e}", "history": conversation_history}

# --- Point d'entrée pour lancer le serveur ---
if __name__ == "__main__":
    # Uvicorn va lancer notre application FastAPI.
    # host="0.0.0.0" le rend accessible depuis l'extérieur de la machine (essentiel sur le cloud)
    uvicorn.run(app, host="0.0.0.0", port=8000)
