import streamlit as st
from openai import OpenAI
import re

client = OpenAI()

# =========================
# 📄 Ler perguntas do TXT
# =========================
def carregar_perguntas(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()

    # Divide por número da pergunta (1., 2., etc.)
    blocos = re.split(r'\n(?=\d+\.\s)', texto)

    perguntas = []

    for bloco in blocos:
        linhas = bloco.strip().split("\n")
        if len(linhas) < 2:
            continue

        pergunta = linhas[0]
        alternativas = linhas[1:]

        perguntas.append({
            "pergunta": pergunta,
            "alternativas": alternativas
        })

    return perguntas


# =========================
# 🤖 Avaliação com GPT
# =========================
def avaliar(pergunta, alternativas, resposta):
    prompt = f"""
Você é um especialista em cervejaria (CIBD).

Pergunta:
{pergunta}

Alternativas:
{chr(10).join(alternativas)}

Resposta do usuário: {resposta}

Tarefa:
1. Diga se está correta ou incorreta
2. Informe a alternativa correta (ex: B)
3. Explique de forma objetiva (máx 4 linhas)
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# 🚀 Streamlit UI
# =========================
st.set_page_config(page_title="Simulado CIBD", layout="centered")

st.title("🍺 Simulado CIBD com IA")

perguntas = carregar_perguntas("perguntas.txt")

# estado
if "indice" not in st.session_state:
    st.session_state.indice = 0

if "resposta" not in st.session_state:
    st.session_state.resposta = None

# pergunta atual
q = perguntas[st.session_state.indice]

st.subheader(f"Pergunta {st.session_state.indice + 1}")

st.write(q["pergunta"])

# extrai letras A, B, C, D
opcoes = [alt[0] for alt in q["alternativas"]]

resposta = st.radio(
    "Escolha sua resposta:",
    options=opcoes,
    format_func=lambda x: next(a for a in q["alternativas"] if a.startswith(x))
)

# botão responder
if st.button("✅ Responder"):
    feedback = avaliar(q["pergunta"], q["alternativas"], resposta)

    st.markdown("### 🧠 Feedback")
    st.write(feedback)

# navegação
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Anterior") and st.session_state.indice > 0:
        st.session_state.indice -= 1

with col2:
    if st.button("➡️ Próxima") and st.session_state.indice < len(perguntas) - 1:
        st.session_state.indice += 1

# progresso
st.progress((st.session_state.indice + 1) / len(perguntas))