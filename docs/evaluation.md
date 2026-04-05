# 📈 Cultura de Datos: Pipeline de Evaluación (Ragas)

En implementaciones serias de RAG Enterprise no basta con "predecir" textos generativos. Implementamos una suite de evaluación heurística con la librería `Ragas v0.2+`, centralizada y envolviendo LangChain con envoltorios oficiales construidos sobre `src.evals.engine`.

## 📐 Métricas Principales

Para testear qué tan bueno es nuestro recuperador y qué tan fiel es nuestra generación, analizamos 2 vertientes puras de Data Science a través de Pandas y dicts:

| Métrica MLOps | Significado Técnico | Tolerancia en Baseline |
| :--- | :--- | :--- |
| **Faithfulness (Fidelidad)** | Mide cuantitativamente qué tan apegada es la repuesta contra el contexto original brindado, previniendo alucinación y controlando respuestas inventadas. | `> 0.60` |
| **Context Precision** | Castiga duramente el ruido en el espacio de representación vectorial. Analiza si los chunks devueltos inyectados fueron oro sólido o simple relleno irrelevante. | `> 0.66` |

## 🏆 El "Golden Dataset"
Nuestra rutina MLOps principal `run_evals.py` cuenta con la dependencia integral orientada a un **Golden Dataset** (Dataset Dorado). Es un sub-grupo compacto de preguntas técnicas y respuestas "Ground Truth" aprobadas por validadores humanos.

Este dataset sirve como un **Test Unitario Estricto** integral para ML. 
- Si nosotros insertamos una modificación drástica en el índice BM25 Léxico que cause "retrievals errantes".
- O actualizamos la librería interna CUDA a 13.0 y rompemos la métrica subyacente de similitud Coseno.

Se ejecutará este bloque de pruebas obligatoriamente para frenar el daño en seco. Cualquier regresión silenciosa que logre degradar la calidad de las respuestas por debajo de nuestro baseline matemático, causará rotunda advertencia a los mantenedores y asegurará una cultura inamovible de excelencia en el ciclo vital de tu IA (*Ready for Production y altamente escalable en QA*).
