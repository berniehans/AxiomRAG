import streamlit as st
import requests
import json
from typing import Dict, Any

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="RAG Explorer - Ops", layout="wide", page_icon="🏛️")

# Estilos simples y pulidos
st.title("🏛️ RAG Enterprise GUI")
st.markdown("*Frontend interactivo. Conecta vía HTTP MLOps al puente `main.py` de FastAPI local.*")

# Tabs Lógicos
tab_chat, tab_ingest = st.tabs(["💬 Laboratorio RAG", "📁 Ingesta Batch"])

# TAB: INGESTA
with tab_ingest:
    st.subheader("Carga Automática a Qdrant")
    uploaded_file = st.file_uploader("Documento corporativo (.pdf, .xlsx, .docx)", type=["pdf", "xlsx", "docx"])
    cat = st.text_input("Override Categoría (Fallback)", "General")
    
    if st.button("Iniciar Extracción de Metadatos (GPU)"):
        if uploaded_file is not None:
            with st.spinner("Fragmentando con BGE-M3 e inyectando metadata vía gpt-4o-mini..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"categoria": cat}
                    r = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=120)
                    if r.status_code == 200:
                        st.success(f"¡{uploaded_file.name} embebido y vectorizado exitosamente en {r.elapsed.total_seconds():.2f}s!")
                    else:
                        st.error(f"Falla Transaccional ({r.status_code}): {r.text}")
                except Exception as e:
                    st.error(f"Problemas de red o backend apagado: {e}")
        else:
            st.warning("Selecciona un documento primero.")

# TAB: CHAT
with tab_chat:
    st.subheader("Interrogatorio Asistido por Computadora")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_input("Consulta técnica estructurada:", placeholder="Ej: ¿Qué algoritmo usa el Semantic Chunker interno?")
        
        if st.button("Ejecutar Cadena"):
            if question:
                with st.spinner("Recuperando Nodos Hijos y Rerankeando vía Cross-Encoder..."):
                    try:
                        r = requests.post(f"{API_URL}/chat", json={"pregunta": question, "session_id": "streamlit-web-1"}, timeout=60)
                        if r.status_code == 200:
                            data = r.json()
                            st.write("### Dictámen de Asistente:")
                            st.info(data["respuesta"])
                            
                            with col2:
                                st.write("### 🔍 Explainer (Reranker Scores)")
                                if not data.get("fuentes"):
                                    st.warning("El Guardrail G-01 interceptó la búsqueda. No hay fuentes disponibles.")
                                else:
                                    for idx, s in enumerate(data.get("fuentes", [])):
                                        scor_color = "normal" if s.get('score', 0) > 0.15 else "inverse"
                                        st.metric(
                                            label=f"📃 {s.get('categoria', 'Uncategorized')} (Ref #{idx})", 
                                            value=f"Score Atencional: {s.get('score', 0):.4f}"
                                        )
                            st.caption(f"Tiempo Total de Operación Pipeline End-to-End: {data.get('tiempo_procesamiento_s', 0):.2f} segundos.")
                        else:
                            st.error(f"Alarma de Endpoint ({r.status_code}): {r.text}")
                    except Exception as e:
                        st.error(f"El Backbone está desconectado o el Timeout LLM se excedió: {e}")
            else:
                st.warning("Pregunta no válida.")
