import requests
from langchain_core.tools import tool

@tool
def convertir_moneda(monto: float, origen: str, destino: str) -> str:
    """Convierte un monto de una moneda a otra usando tasas de cambio actuales. 
    Recibe el monto, el código de moneda de origen (ej: USD) y el código de moneda destino (ej: EUR).
    Soporta principalmente monedas fuertes (USD, EUR, GBP, JPY, etc.), no incluye monedas latinoamericanas como ARS."""
    url = "https://api.frankfurter.dev/v1/latest"
    params = {
        "amount": monto,
        "base": origen.upper(),
        "symbols": destino.upper()
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "rates" not in data or destino.upper() not in data["rates"]:
        return f"No se pudo convertir de {origen} a {destino}. Verificá que ambos códigos de moneda sean válidos (ej: USD, EUR, GBP, JPY)."

    resultado = data["rates"][destino.upper()]
    return f"{monto} {origen.upper()} equivalen a {resultado} {destino.upper()} (tasa del {data['date']})."


if __name__ == "__main__":
    print(convertir_moneda.invoke({"monto": 100, "origen": "USD", "destino": "EUR"}))