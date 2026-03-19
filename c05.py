# pip install pydantic-ai pytrends python-dotenv pandas

import time
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

load_dotenv()

HEADERS = {"User-Agent": "ConsumerInsightsBot/1.0"}

# contador de chamadas
contador = {"n": 0}


# =========================
# TOOL: Google Trends
# =========================

def pesquisar_trends(termo: str) -> str:
    """
    Busca um termo no Google Trends e retorna queries relacionadas.
    Use para entender associações de consumo e contexto de marca.
    Sempre pesquise UM termo por vez.
    """
    contador["n"] += 1
    print(f"  [Tool call #{contador['n']}] buscando '{termo}'...")

    pytrends = TrendReq(hl="pt-BR", tz=-180)

    try:
        pytrends.build_payload([termo], timeframe="today 12-m", geo="BR")
        related = pytrends.related_queries()

        bloco = related.get(termo, {})
        top = bloco.get("top")
        rising = bloco.get("rising")

        saida = [f"[Termo: {termo}]"]

        if top is not None and not top.empty:
            saida.append("Top queries:")
            for _, row in top.head(5).iterrows():
                saida.append(f"- {row['query']} ({row['value']})")
        else:
            saida.append("Top queries: nenhum dado")

        if rising is not None and not rising.empty:
            saida.append("Rising queries:")
            for _, row in rising.head(5).iterrows():
                saida.append(f"- {row['query']} ({row['value']})")
        else:
            saida.append("Rising queries: nenhum dado")

        print(f"  [#{contador['n']} concluído — termo: {termo}]")

        time.sleep(1)  # evita bloqueio
        return "\n".join(saida)

    except TooManyRequestsError:
        return f"[Termo: {termo}] Erro 429 (muitas requisições)"

    except Exception as e:
        return f"[Termo: {termo}] Erro: {e}"


# =========================
# AGENTE
# =========================

agente = Agent(
    model=OpenRouterModel("openai/gpt-4o-mini"),
    tools=[pesquisar_trends],
    system_prompt=(
        "Você é um analista de Consumer Insights focado em marcas e comportamento do consumidor.\n\n"
        
        "Seu objetivo é entender COM O QUE uma marca combina na vida do consumidor.\n\n"

        "REGRAS:\n"
        "- Sempre use a ferramenta de Google Trends\n"
        "- Pesquise cada termo separadamente\n"
        "- Comece pela marca principal\n"
        "- Depois explore termos relacionados (ex: ocasião, atributo, contexto)\n"
        "- Use no máximo 4 buscas\n\n"

        "Depois das buscas, analise:\n"
        "- associações de consumo\n"
        "- ocasiões\n"
        "- território da marca\n"
        "- combinações típicas\n\n"

        "Responda como um analista de insights (não como técnico)."
    ),
)


# =========================
# EXECUÇÃO
# =========================

print("=" * 60)
print("Radar simples de território de marca (ReAct)")
print("=" * 60)

pergunta = input("\nDigite sua pergunta: ")

print("\nExecutando análise...\n")

result = agente.run_sync(pergunta)

print("\n" + "=" * 60)
print("RESPOSTA FINAL:")
print("=" * 60)
print(result.output)

print("\n" + "=" * 60)
print(f"Tool calls executados: {contador['n']}")
print(f"Chamadas ao modelo: {result.usage().requests}")