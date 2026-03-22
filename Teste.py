from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# (Opcional) LLM via OpenRouter
try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openrouter import OpenRouterModel
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Insights Comercial", layout="wide")

st.title("📊 Insights Comercial – Causalidade de Demanda")

# ========================
# FILTROS
# ========================
st.sidebar.header("Filtros")

estado = st.sidebar.selectbox("Estado", ["SP", "RJ", "MG"])
cidade = st.sidebar.selectbox("Cidade", ["São Paulo", "Campinas", "Rio de Janeiro"])
categoria = st.sidebar.selectbox("Categoria", ["Cerveja", "Refrigerante"])
marca = st.sidebar.selectbox("Marca", ["Brahma", "Skol", "Guaraná"])
sku = st.sidebar.selectbox("SKU", ["SKU 1", "SKU 2", "SKU 3"])
periodo = st.sidebar.date_input("Período", [datetime(2024,1,1), datetime(2024,1,30)])

# ========================
# DADOS SIMULADOS
# ========================
dates = pd.date_range(start="2024-01-01", periods=30)
previsto = np.random.randint(80, 120, size=30)
realizado = previsto + np.random.randint(-20, 20, size=30)

df = pd.DataFrame({
    "Data": dates,
    "Previsto": previsto,
    "Realizado": realizado
})

# ========================
# GRÁFICO
# ========================
st.subheader("📈 Demanda Prevista vs Realizada")
st.line_chart(df.set_index("Data"))

# ========================
# DISPERSÃO
# ========================
dispersao = ((df["Realizado"].sum() - df["Previsto"].sum()) / df["Previsto"].sum()) * 100
st.metric("Dispersão (%)", f"{dispersao:.2f}%")

# ========================
# RANKING DE CAUSAS
# ========================
st.subheader("🧩 Principais Causas")

causas = pd.DataFrame({
    "Fator": ["Ruptura", "Clima", "Preço Concorrente", "Execução Trade", "Social Buzz"],
    "Impacto (%)": [-18, -7, -9, 5, 11]
})

st.bar_chart(causas.set_index("Fator"))

# ========================
# TABELA DETALHADA
# ========================
st.subheader("📊 Detalhamento de Fatores")

tabela = pd.DataFrame({
    "Variável": causas["Fator"],
    "Tipo": ["Interno", "Externo", "Concorrência", "Interno", "Externo"],
    "Impacto (%)": causas["Impacto (%)"],
    "Direção": ["↓", "↓", "↓", "↑", "↑"],
    "Confiança": np.round(np.random.uniform(0.7, 0.95, 5), 2)
})

st.dataframe(tabela)

# ========================
# EXPLORADOR DE EFEITOS
# ========================
st.subheader("🔍 Explorar Efeitos")

fator_escolhido = st.selectbox("Selecione um fator para detalhar", causas["Fator"])

if fator_escolhido == "Social Buzz":
    st.write("### 🌐 Varredura de Menções em Redes Sociais (Simulado)")
    social_df = pd.DataFrame({
        "Data": dates,
        "Menções": np.random.randint(50, 200, size=30),
        "Sentimento": np.random.choice(["Positivo", "Neutro", "Negativo"], size=30)
    })
    st.line_chart(social_df.set_index("Data")["Menções"])
    st.write("Exemplo: aumento de menções positivas correlacionado com aumento de vendas.")

with st.expander("Ver mais detalhes do efeito"):
    st.write(f"O fator {fator_escolhido} apresentou impacto relevante na demanda.")

# ========================
# HIPÓTESES
# ========================
st.subheader("🧪 Hipóteses Geradas")

hipoteses = pd.DataFrame({
    "Hipótese": [
        "Alta ruptura reduziu vendas",
        "Queda de temperatura impactou consumo",
        "Aumento de buzz positivo elevou demanda"
    ],
    "Confiança Estatística": [0.91, 0.85, 0.88]
})

st.dataframe(hipoteses)

# ========================
# INSIGHTS (LLM OU MOCK)
# ========================
st.subheader("🧠 Insights")

prompt_insight = f"""
Explique a dispersão de demanda de {dispersao:.1f}% considerando:
{tabela.to_dict()}
"""

if LLM_AVAILABLE:
    agente = Agent(
        model=OpenRouterModel("openai/gpt-4o-mini"),
        system_prompt="Você é um analista que explica causalidade de demanda sem prever futuro."
    )
    resposta = agente.run_sync(prompt_insight)
    st.write(resposta.output)
else:
    st.write("Simulação: Ruptura e concorrência foram os principais drivers.")

# ========================
# CHAT
# ========================
st.subheader("💬 Chat com IA")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Pergunte algo sobre os dados")

if user_input:
    st.session_state.chat_history.append({"user": user_input})

    if LLM_AVAILABLE:
        resposta = agente.run_sync(user_input)
        resposta_texto = resposta.output
    else:
        resposta_texto = "Resposta simulada baseada nos dados exibidos."

    st.session_state.chat_history.append({"bot": resposta_texto})

for msg in st.session_state.chat_history:
    if "user" in msg:
        st.write(f"👤 {msg['user']}")
    else:
        st.write(f"🤖 {msg['bot']}")