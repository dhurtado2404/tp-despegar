# TP: Agente de viajes con LangChain + ReAct

Se implementa un agente conversacional de una agencia de viajes usando **LangChain + Ollama** con el patrón **ReAct**.

---

## 1. Fundamentos teóricos

### 1.1 ¿Qué es un agente?

Un LLM "puro" responde preguntas, pero no puede ejecutar acciones en el mundo. Un **agente** es un sistema que envuelve a un LLM y le da la capacidad de:

- **Razonar** sobre lo que el usuario pide
- **Elegir herramientas (tools)** para conseguir información o ejecutar acciones
- **Observar** el resultado de cada tool
- **Iterar** hasta tener una respuesta satisfactoria

A diferencia de un simple RAG (que solo recupera documentos), un agente puede encadenar múltiples acciones en distintas órdenes según lo que necesite.

### 1.2 El patrón ReAct (Reasoning + Acting)

ReAct es un patrón introducido por [Yao et al. (2022)](https://arxiv.org/abs/2210.03629) que alterna dos tipos de "pasos":

- **Thought (razonamiento)**: el modelo piensa qué necesita hacer a continuación.
- **Action (acción)**: el modelo invoca una tool con un input.
- **Observation (observación)**: la tool devuelve un resultado, que se inyecta de vuelta al modelo.

El ciclo se repite hasta que el modelo emite un `Final Answer`.

```
Question: ¿Cuánto sale un viaje a Madrid?
Thought: Necesito buscar vuelo y hotel.
Action: buscar_vuelo → V002 (USD 850)
Action: buscar_hotel → H008 (USD 40/noche)
Action: calcular_paquete → USD 1250
Final Answer: El paquete cuesta USD 1250.
```

**Alternativa moderna**: modelos con *tool calling* nativo (GPT-4, Claude, llama3.1+) usan un esquema JSON estructurado en vez de texto libre. En este TP usamos ReAct por prompt porque `gemma3` (4B) no soporta tool calling nativo.

### 1.3 Tools en LangChain

Una *tool* es una función Python con un **docstring claro**. LangChain usa ese docstring para que el LLM entienda:
- Qué hace la tool
- Qué argumentos espera
- Qué devuelve

```python
@tool
def buscar_vuelo(consulta: str) -> str:
    """Busca vuelos disponibles. Args: consulta en lenguaje natural."""
    ...
```

### 1.4 RAG como tool

Una de las tools implementa **Retrieval-Augmented Generation (RAG)** sobre las políticas de la agencia:

1. **Indexado**: las 6 políticas se convierten en embeddings con `nomic-embed-text` (Ollama) y se guardan en **ChromaDB** (en memoria).
2. **Recuperación**: cuando el usuario pregunta algo, se hace *similarity search* y se devuelven los top-k documentos relevantes.
3. **Generación**: el LLM usa esos documentos para formular la respuesta final.

Esto evita que el modelo "alucine" sobre políticas que no conoce.

### 1.5 Memoria conversacional

El agente mantiene una ventana de los últimos `MAX_TURNOS=10` turnos (par pregunta-respuesta). En cada llamada, el historial se inyecta en el prompt para que el agente entienda seguimientos del estilo *"y si le sumo 2 noches más?"*.

---

## 2. Arquitectura

```
                 ┌────────────────────────────┐
   pregunta ───► │  AGENTE (ReAct, gemma3)    │
                 │  + historial de la charla  │
                 └────────────┬───────────────┘
        piensa / elige tool   │   observa el resultado y repite
        ┌─────────┬───────────┼───────────┬──────────────┐
        ▼         ▼           ▼           ▼              ▼
   buscar_vuelo buscar_hotel calcular_   consultar_
                              paquete      politicas (RAG)
        └─────────┴───────────┴───────────┴──────────────┘
                              ▼
                        Final Answer
```

**Stack:**
- **LLM**: `gemma3` (4B, corre local con Ollama)
- **Embeddings**: `nomic-embed-text` (local)
- **Framework**: LangChain (`create_react_agent`, `AgentExecutor`)
- **Vector DB**: ChromaDB (en memoria)
- **Observabilidad**: LangSmith (opcional)

---

## 3. Estructura del proyecto

```
tp_despegar/
├── datos/
│   ├── vuelos.json       # 8 vuelos mock
│   ├── hoteles.json      # 8 hoteles mock
│   └── politicas.json    # 6 políticas (texto para RAG)
├── tools/
│   ├── __init__.py
│   ├── vuelos.py         # tool: buscar_vuelo
│   ├── hoteles.py        # tool: buscar_hotel
│   ├── paquete.py        # tool: calcular_paquete
│   └── politicas.py      # tool: consultar_politicas (RAG)
├── agente.py             # agente ReAct + memoria + LangSmith
├── tp_agente_despegar.ipynb  # notebook entregable
├── README.md             # este archivo
├── requirements.txt
└── .env.example
```

---

## 4. Instalación y ejecución

### Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) instalado y corriendo

### Pasos

```bash
# 1. Clonar/copiar la carpeta tp_despegar/
cd tp_despegar

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelos de Ollama (solo la primera vez)
ollama pull gemma3
ollama pull nomic-embed-text

# 4. (Opcional) Configurar LangSmith para ver las trazas en la nube
cp .env.example .env
# Editar .env y completar LANGSMITH_API_KEY

# 5. Correr el agente
python agente.py                              # modo interactivo
python agente.py "Viaje a Madrid del 15 al 25 de septiembre"  # una sola pregunta
```

### Uso del notebook

Abrir `tp_agente_despegar.ipynb` en VS Code o Jupyter y ejecutar celda por celda. Las celdas de Markdown explican la arquitectura, los 3 ejemplos del TP y el análisis crítico.

---

## 5. Ejemplos de uso

```text
> Quiero viajar a Madrid del 15 al 25 de septiembre, armame un paquete completo
[Thought] -> buscar_vuelo -> V002 (USD 850)
[Thought] -> buscar_hotel -> H008 (USD 40)
[Thought] -> calcular_paquete -> USD 1250
Final Answer: El paquete cuesta USD 1250.

> Necesito visa para ir a España y cuánto equipaje puedo llevar?
[Thought] -> consultar_politicas (k=2) -> [visa, equipaje]
Final Answer: Visa Schengen + equipaje en bodega incluido.
```

---

## 6. Análisis crítico: limitaciones observadas

Durante el desarrollo se detectaron 3 limitaciones reales:

### 6.1 Loops infinitos en inputs ambiguos

**Caso**: input `"hoy"` (sin contexto).
**Síntoma**: el agente invoca `buscar_vuelo("quiero un vuelo a Madrid")` 8 veces hasta que el executor corta por `max_iterations`.
**Causa**: el modelo no interpreta la observación "necesito destino" como señal de "preguntale al usuario".
**Mejora**: agregar una tool `preguntar_usuario` y migrar a un modelo con tool-calling nativo (ej. `llama3.1`).

### 6.2 Filtros no implementados en tools

**Caso**: *"Y si en vez de hostel me das un hotel de 4 estrellas?"*.
**Síntoma**: el agente llama `buscar_hotel("hotel en Madrid 4 estrellas")` 8 veces, recibiendo siempre H008 (2 estrellas).
**Causa**: el parser de la tool no extrae "estrellas" como filtro.
**Mejora**: extender el parser + regla en el prompt *"si la observación no refleja tu pedido, reformulá"*.

### 6.3 Comportamiento errático ante datos faltantes

**Caso**: *"Quiero ir a Tokio en enero"*.
**Síntoma**: tras detectar "no hay vuelos", el agente llama innecesariamente a `consultar_politicas("puedo buscar otro destino")`.
**Causa**: el prompt instruye "resolvé cada parte con su propia Action" y el modelo no sabe cuándo cortar.
**Mejora**: regla explícita *"si una tool devuelve 'no encontrado', pasá directo a Final Answer"*.

---

## 7. Posibles mejoras

| Mejora | Impacto |
|--------|---------|
| Migrar a modelo con tool-calling nativo (`llama3.1`) | Reduce loops, formato robusto |
| Persistir historial en SQLite | Sobrevive reinicios |
| Conectar a APIs reales (Amadeus, Skyscanner) | Aplicable a producción |
| Tool `preguntar_usuario` para clarificar | Resuelve Limitación 6.1 |
| Filtro de estrellas en `buscar_hotel` | Resuelve Limitación 6.2 |
| Validación de input mínima antes de iterar | Resuelve Limitación 6.1 |

---

## 8. Observabilidad (opcional)

Con `LANGSMITH_API_KEY` en `.env`, cada corrida se registra en [smith.langchain.com](https://smith.langchain.com) bajo el proyecto `tp-despegar-agente`. Permite ver el árbol completo `Thought → Action → Observation` con latencias y tokens. **No es obligatorio** para que el agente funcione.

---

## 9. Referencias

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)
- [LangChain Agents docs](https://python.langchain.com/docs/modules/agents/)
- [Ollama](https://ollama.com)
- [ChromaDB](https://www.trychroma.com)
- Material del curso: `agente/agente_completo.py` (versión de referencia)
