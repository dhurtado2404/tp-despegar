from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools.clima import obtener_clima
from tools.moneda import convertir_moneda

# 1. El modelo
llm = ChatOllama(model="qwen2.5:7b", temperature=0.3, repeat_penalty=1.3)

# 2. Lista de tools disponibles para el agente
tools = [obtener_clima, convertir_moneda]

# 3. Armamos el agente (LangGraph maneja todo el ciclo ReAct por dentro)
agent = create_agent(llm, tools, system_prompt="Respondé siempre en español, de forma clara y concisa.")

# 4. Lo invocamos con una pregunta
if __name__ == "__main__":
    print("Agente listo. Escribí 'salir' para terminar.\n")
    while True:
        pregunta = input("Vos: ")
        if pregunta.lower() in ("salir", "exit", "quit"):
            break
        
        response = agent.invoke({"messages": [("user", pregunta)]})
        
        for mensaje in response["messages"]:
            mensaje.pretty_print()
        
        print()