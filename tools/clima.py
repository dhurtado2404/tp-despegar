import requests
from langchain_core.tools import tool



def obtener_coordenadas(ciudad: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": ciudad}
    response = requests.get(url, params=params)
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        return None  # no se encontró la ciudad
    
    primer_resultado = data["results"][0]
    lat = primer_resultado["latitude"]
    lon = primer_resultado["longitude"]
    nombre_completo = f"{primer_resultado['name']}, {primer_resultado['country']}"
    
    return lat, lon, nombre_completo

def obtener_clima_por_coordenadas(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

WEATHER_CODES = {
    0: "despejado",
    1: "mayormente despejado",
    2: "parcialmente nublado",
    3: "nublado",
    45: "neblina",
    51: "llovizna ligera",
    61: "lluvia ligera",
    63: "lluvia moderada",
    65: "lluvia fuerte",
    71: "nieve ligera",
    80: "chubascos",
    95: "tormenta",
}

def describir_clima(codigo: int) -> str:
    return WEATHER_CODES.get(codigo, "condición desconocida")


@tool
def obtener_clima(ciudad: str) -> str:
    """Obtiene el clima actual de una ciudad. Recibe el nombre de la ciudad como parámetro."""
    resultado = obtener_coordenadas(ciudad)
    if resultado is None:
        return f"No se encontró la ciudad '{ciudad}'."
    
    lat, lon, nombre_completo = resultado
    data = obtener_clima_por_coordenadas(lat, lon)
    
    temp = data["current_weather"]["temperature"]
    viento = data["current_weather"]["windspeed"]
    codigo = data["current_weather"]["weathercode"]
    condicion = describir_clima(codigo)
    
    return f"En {nombre_completo}: {temp}°C, {condicion}, viento de {viento} km/h."

if __name__ == "__main__":
    print(obtener_clima.invoke({"ciudad": "Buenos Aires"}))
