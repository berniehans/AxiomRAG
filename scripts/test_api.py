import requests
import time
import json
import sys
import os
import shutil

# Agregado fallback local por si falla el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Asegurar import settings para ver paths
URL = "http://127.0.0.1:8000"

def run_tests():
    print(f"📡 Testeando API contra {URL} 📡")
    
    # 1. Endpoint /ingest Upload con validaciones fuertes
    print("\n--- 1. [POST] /ingest Simulación ----")
    
    source_pdf = os.path.join("data", "sample.pdf")
    test_pdf = os.path.join("data", "test_ingesta_web.pdf")
    
    # Validación de Origen
    if not os.path.exists(source_pdf):
        raise FileNotFoundError(f"Se requiere un PDF real en la ruta maestra: {source_pdf}")
    
    # Verificación de Integridad Binaria
    with open(source_pdf, "rb") as f:
        header = f.read(4)
        if header != b"%PDF":
            raise ValueError(f"El archivo {source_pdf} está corrupto y no es un PDF nativo.")
            
    # Clonación Segura
    shutil.copy2(source_pdf, test_pdf)
        
    try:
        with open(test_pdf, "rb") as f:
            files = {"file": (os.path.basename(test_pdf), f, "application/pdf")}
            res = requests.post(f"{URL}/ingest", files=files, data={"categoria": "Investigación IA"}, timeout=300)
            print(f"Estatus HTTP: {res.status_code}\nCuerpo: {res.json()}")
    except Exception as e:
        print(f"Falla crítica: servidor apagado o error de red: {e}")
        print("Asegúrate de tener un terminal corriendo: 'uv run uvicorn src.main:app --reload'")
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        return
        
    # Limpieza Post-Test
    if os.path.exists(test_pdf):
        os.remove(test_pdf)

    # 2. Endpoint /chat Stateful
    print("\n--- 2. [POST] /chat (QueryRequest) ----")
    session = "token_de_usuario_api_88"
    
    questions = [
        "¿Cuál es la arquitectura principal propuesta en el paper y por qué prescinde de las redes recurrentes (RNN)?",
        "Explica detalladamente cómo funciona el mecanismo de Multi-Head Attention según el documento.",
        "¿Cuál es la capital de Francia?"
    ]
    
    for q in questions:
        print(f"\n🧑‍💻 JSON Payload: { {'pregunta': q, 'session_id': session} }")
        t_start = time.time()
        
        try:
             chat_req = requests.post(
                 f"{URL}/chat", 
                 json={"pregunta": q, "session_id": session},
                 timeout=300
             )
             if chat_req.status_code == 200:
                  data = chat_req.json()
                  print(f"💻 API RAG Answer: {data['respuesta']}")
                  print(f"⏱️ Tiempo Latencia LLM: {data['tiempo_procesamiento_s']}s")
                  if data.get('fuentes'):
                      print(f"🎯 Score Top 1 (Reranker): {data['fuentes'][0]['score']:.4f}")
                      print(f"🕵️‍♂️ Citation Logs: {json.dumps(data['fuentes'], indent=2)}")
                  else:
                      print("🛑 No se recuperaron fuentes viables por el guardrail.")
             else:
                  print(f"🛑 Fail. {chat_req.text}")
        except Exception as e:
             print(f"Exception local: {e}")

if __name__ == "__main__":
    run_tests()
