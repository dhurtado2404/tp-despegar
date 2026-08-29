# Compass — Asistente de viajes con Agentes LLM

Trabajo Práctico: Aplicaciones con Modelos de Lenguaje (LLMs) — Opción B: Agentes

## 1. Descripción general

Compass es un agente conversacional basado en LLMs capaz de responder consultas relacionadas con planificación de viajes, combinando razonamiento del modelo con herramientas (*tools*) que consultan información externa en tiempo real. El agente decide de forma autónoma qué herramienta usar según la pregunta del usuario, siguiendo el ciclo de razonamiento **ReAct** (Reason + Act).

## 2. Fundamentos teóricos

### 2.1 ¿Qué es un agente LLM?

Un LLM por sí solo solo puede generar texto: no puede ejecutar código, consultar una API ni acceder a datos externos. Un **agente** es un sistema que usa un LLM como "motor de razonamiento", combinado con **herramientas (tools)** que sí pueden ejecutar acciones concretas. El agente decide, en base al lenguaje natural de la consulta, qué herramienta invocar, con qué parámetros, y cómo usar el resultado para construir una respuesta.

### 2.2 Tool calling

Es la capacidad, entrenada específicamente en ciertos modelos, de generar una salida estructurada (JSON) que representa la intención de invocar una función con determinados argumentos, en lugar de responder directamente en texto libre. Ese JSON no lo ejecuta el modelo: lo interpreta el framework (en este caso, LangChain/LangGraph), que llama a la función real en Python y devuelve el resultado al modelo.

### 2.3 Ciclo ReAct (Reason + Act)

El patrón ReAct intercala pasos de razonamiento y acción:

1. **Pensar**: el modelo analiza la consulta y decide si necesita información externa.
2. **Actuar**: si es necesario, invoca una tool con los parámetros correspondientes.
3. **Observar**: recibe el resultado real de la tool (no inventado).
4. **Responder**: integra la observación en una respuesta final en lenguaje natural, o repite el ciclo si necesita más información.

### 2.4 LangChain vs. LangGraph

**LangChain** provee las piezas básicas (modelos, tools, prompts) pensadas originalmente para flujos lineales ("chains"). **LangGraph** modela el comportamiento de un agente como un **grafo de estados** con ciclos y decisiones condicionales, lo cual representa de forma más natural el ciclo ReAct (pensar → actuar → observar → volver a pensar o responder). Este proyecto usa `create_agent` (LangChain, apoyado en LangGraph por debajo), que es el enfoque actualmente recomendado por la documentación oficial para construir agentes, en reemplazo del `AgentExecutor` clásico.

## 3. Arquitectura del sistema

- **Modelo**: `qwen2.5:7b`, ejecutado localmente vía Ollama, elegido por su soporte nativo de tool calling y por permitir ejecución 100% local sin costo de API.
- **Framework de orquestación**: LangChain (`create_agent`) + LangGraph.
- **Herramientas (tools)**:
  - `obtener_clima(ciudad)`: consulta el clima actual de una ciudad vía la API pública [Open-Meteo](https://open-meteo.com/) (geocoding + clima), sin necesidad de API key.
  - `convertir_moneda(monto, origen, destino)`: convierte montos entre monedas usando tasas de cambio actuales de la API [Frankfurter](https://frankfurter.dev/), sin necesidad de API key. Soporta monedas "fuertes" (USD, EUR, GBP, JPY, etc.); no incluye monedas latinoamericanas como ARS.


## 4. Instalación y ejecución

### Requisitos previos

- Python 3.10+
- [Ollama](https://ollama.com) instalado y corriendo localmente
- Modelo `qwen2.5:7b` descargado:
  ```bash
  ollama pull qwen2.5:7b
  ```

### Setup del entorno

```bash
git clone https://github.com/dhurtado2404/tp-despegar.git
cd tp-despegar
python3 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecución

Opción 1 — Modo script interactivo:
```bash
python app.py
```

Opción 2 — Notebook:
```bash
jupyter notebook
```
Abrir `agente_viajes.ipynb` y ejecutar las celdas en orden.

## 5. Ejemplos de uso

*(completar con las 3 trazas ReAct finales, copiadas del notebook — incluir al menos: una pregunta que use solo `obtener_clima`, una que use solo `convertir_moneda`, y una que combine ambas tools en una sola consulta)*

### Ejemplo 1: Consulta de clima
```
================================ Human Message =================================
¿Qué clima hace en Madrid?
================================== Ai Message ==================================
Tool Calls:
  obtener_clima (2ba329f6-5547-471b-9207-b5f0b3712988)
 Call ID: 2ba329f6-5547-471b-9207-b5f0b3712988
  Args:
    ciudad: Madrid
================================= Tool Message =================================
Name: obtener_clima
En Madrid, Spain: 16.9°C, despejado, viento de 4.0 km/h.
================================== Ai Message ==================================
Actualmente en Madrid hace 16.9 grados Celsius con condiciones despejadas, el viento está soplando a una velocidad de 4.0 kilómetros por hora.
```

### Ejemplo 2: Conversión de moneda
```
================================ Human Message =================================
¿Cuánto equivalen 100 USD a euros?
================================== Ai Message ==================================
Tool Calls:
  convertir_moneda (dda22220-7cad-450b-835e-316c3073143b)
 Call ID: dda22220-7cad-450b-835e-316c3073143b
  Args:
    monto: 100
    origen: USD
    destino: EUR
================================= Tool Message =================================
Name: convertir_moneda
100.0 USD equivalen a 85.89 EUR (tasa del 2026-08-28).
================================== Ai Message ==================================
100 USD equivalen a aproximadamente 85.89 Euros, según la tasa actual de cambio al día 28/08/2026.
```

### Ejemplo 3: Consulta combinada
```
================================ Human Message =================================
¿Qué clima hace en Madrid y cuánto son 100 dólares en euros?
================================== Ai Message ==================================
Tool Calls:
  obtener_clima (acca0b67-62ec-4568-b231-7489bea7d9b8)
 Call ID: acca0b67-62ec-4568-b231-7489bea7d9b8
  Args:
    ciudad: Madrid
  convertir_moneda (94116487-137c-484b-97ce-f71aab526e25)
 Call ID: 94116487-137c-484b-97ce-f71aab526e25
  Args:
    monto: 100
    origen: USD
    destino: EUR
================================= Tool Message =================================
Name: obtener_clima
En Madrid, Spain: 16.6°C, despejado, viento de 4.0 km/h.
================================= Tool Message =================================
Name: convertir_moneda
100.0 USD equivalen a 85.89 EUR (tasa del 2026-08-28).
================================== Ai Message ==================================
En Madrid hace un clima de 16.6°C con condiciones despejadas, y el viento es de 4.0 km/h.
Además, 100 USD equivalen a aproximadamente 85.89 EUR según la tasa actual del día 28/08/2026.
```

## 6. Análisis crítico

### Limitaciones actuales

- El agente no mantiene memoria entre invocaciones independientes en el modo script (cada pregunta se procesa sin contexto de las anteriores).
- La conversión de moneda depende de la API Frankfurter, que no soporta monedas latinoamericanas (ej. ARS).
- El geocoding de ciudades homónimas (ej. "Buenos Aires" existe en varios países) se resuelve tomando el primer resultado devuelto por la API.
- Al ser un modelo de 7B parámetros ejecutado localmente, el razonamiento es menos robusto que el de modelos más grandes (GPT-4, Claude, etc.); en algunos casos se observaron repeticiones o mezcla de idiomas en la generación, mitigado ajustando `temperature` y `repeat_penalty`.

### Posibles mejoras

- Agregar memoria de conversación entre turnos usando un `checkpointer` de LangGraph, para permitir preguntas de seguimiento contextuales.
- Incorporar una tool adicional para desambiguar ciudades homónimas, pidiendo confirmación del país al usuario.
- Evaluar el uso de un modelo más grande (o vía API) para consultas que requieran mayor precisión de razonamiento.
