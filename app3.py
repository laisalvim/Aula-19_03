from __future__ import annotations

import os
import json
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

load_dotenv()

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="AI Architect Multi-Agentes",
    page_icon="🧠",
    layout="wide",
)

# =========================================================
# CONFIGURAÇÃO DO MODELO
# =========================================================
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.warning(
        "OPENROUTER_API_KEY não encontrada no ambiente. "
        "Adicione a variável no .env para o app funcionar."
    )

modelo = OpenRouterModel(MODEL_NAME)

# =========================================================
# SCHEMAS
# =========================================================

class ToolSugerida(BaseModel):
    nome: str = Field(description="Nome da ferramenta sugerida")
    finalidade: str = Field(description="Para que a ferramenta serve no contexto do agente")
    tipo: str = Field(description="Ex.: search, database, api, file, web, internal, validation, workflow")
    prioridade: str = Field(description="Alta, Média ou Baixa")


class ArquiteturaAvaliacao(BaseModel):
    nome: str
    como_funciona_no_contexto: str
    pros: List[str]
    contras: List[str]
    adequacao: str = Field(description="Alto, Médio ou Baixo")


class AgenteDefinido(BaseModel):
    nome: str
    responsabilidade: str
    ferramentas_sugeridas: List[ToolSugerida]


class FluxoPasso(BaseModel):
    ordem: int
    descricao: str


class PlanoArquiteturaMultiAgentes(BaseModel):
    entendimento_problema: str
    premissas: List[str]
    avaliacoes: List[ArquiteturaAvaliacao]
    arquitetura_principal: str
    justificativa_escolha: str
    arquitetura_secundaria: Optional[str] = None
    quantidade_de_agentes: int
    agentes: List[AgenteDefinido]
    fluxo_operacional: List[FluxoPasso]
    observacoes_execucao: List[str]


class CodigoOrquestrador(BaseModel):
    nome_arquivo: str = Field(description="Nome sugerido do arquivo")
    codigo_python: str = Field(description="Código completo do orquestrador")
    explicacao: str = Field(description="Resumo do que o código faz")
    dependencias: List[str] = Field(description="Dependências principais para executar")
    proximos_passos: List[str] = Field(description="Passos para conectar o código ao projeto real")


# =========================================================
# AGENTE 1: ORQUESTRADOR DE ARQUITETURA
# =========================================================

agente_orquestrador = Agent(
    model=modelo,
    output_type=PlanoArquiteturaMultiAgentes,
    system_prompt="""
Você é um AI Architect especializado em sistemas multi-agentes.

Sua função é analisar um caso de uso descrito em linguagem natural e projetar a melhor arquitetura possível.

Você DEVE seguir este processo rigoroso:

1. Entender profundamente o problema.
2. Avaliar explicitamente as seguintes arquiteturas:
   - Prompt Chaining
   - Parallelization
   - Routing
   - Orchestrator-Workers

3. Para cada arquitetura, você deve:
   - Explicar como funcionaria no contexto
   - Avaliar prós e contras
   - Dar um nível de adequação: Alto, Médio ou Baixo

4. Escolher uma arquitetura principal.
   - Não seja ambíguo
   - Justifique claramente a escolha
   - Pode citar uma secundária, mas sem indecisão

5. Definir a estrutura de agentes:
   - Quantidade de agentes
   - Nome de cada agente
   - Responsabilidade clara
   - Ferramentas sugeridas para cada agente

6. Descrever o fluxo operacional:
   - Passo a passo de execução
   - Como os agentes interagem

Regras importantes:
- O usuário irá descrever o caso de uso em linguagem natural
- Você deve interpretar, estruturar e propor a arquitetura
- Não peça que o usuário reformule em formato técnico
- Não pule nenhuma arquitetura
- Seja analítico e crítico, não superficial
- Estruture a resposta como um plano executivo
- Seja claro, direto e decisivo
- Sempre finalize com uma arquitetura principal única e definitiva
- Ao sugerir ferramentas, pense em ferramentas reais ou classes de ferramentas úteis para cada agente
""",
)

# =========================================================
# AGENTE 2: GERADOR DE CÓDIGO DO ORQUESTRADOR
# =========================================================

agente_codigo = Agent(
    model=modelo,
    output_type=CodigoOrquestrador,
    system_prompt="""
Você é um engenheiro de software especialista em Pydantic AI e em arquiteturas multi-agentes.

Sua tarefa é gerar o código do orquestrador com base na arquitetura final escolhida.

Você receberá:
- o caso de uso
- o plano da arquitetura
- a arquitetura principal escolhida
- a estrutura dos agentes
- o fluxo operacional

OBJETIVO:
Gerar um código Python completo, organizado e executável, usando Pydantic AI de forma correta e coerente com a recomendação final.

REGRAS CRÍTICAS:
- Você DEVE usar pydantic_ai.Agent para definir os agentes
- Você NÃO deve criar classes Python simples para simular agentes
- Você NÃO deve sugerir ferramentas
- Você NÃO deve incluir listas de ferramentas, integrações ou capacidades externas
- Você deve focar apenas nas funções dos agentes e na articulação entre eles
- O orquestrador deve coordenar os agentes por meio de chamadas explícitas
- O código deve refletir fielmente a arquitetura principal escolhida
- O código deve ser consistente com o fluxo operacional definido no plano
- O código pode ser um esqueleto funcional, mas deve ser tecnicamente correto e coerente
- Inclua schemas Pydantic apenas quando forem necessários para estruturar as saídas
- Use nomes claros para agentes, funções e etapas do fluxo
- Não escreva texto explicativo dentro do código além de comentários curtos e úteis
- Priorize clareza, execução e aderência arquitetural

O QUE O CÓDIGO DEVE CONTER:
- imports necessários
- configuração do modelo
- definição dos agentes com Agent(...)
- schemas de saída com BaseModel, se necessário
- função principal do orquestrador
- articulação entre os agentes conforme a arquitetura escolhida
- bloco de execução principal, se fizer sentido

FORMATO DA RESPOSTA:
Entregue:
- nome_arquivo
- codigo_python
- explicacao
- dependencias
- proximos_passos

INSTRUÇÕES DE QUALIDADE:
- Gere código que pareça pronto para adaptação em um projeto real
- Não improvise abstrações desnecessárias
- Não saia do escopo da arquitetura escolhida
- Não use classes tradicionais para representar agentes
- Não sugira ferramentas
- Não misture responsabilidades entre agentes
""",
)

# =========================================================
# FUNÇÕES PRINCIPAIS
# =========================================================

def projetar_arquitetura(caso_de_uso: str) -> PlanoArquiteturaMultiAgentes:
    prompt = f"""
Analise o caso de uso abaixo e proponha uma arquitetura multi-agentes completa.

CASO DE USO:
{caso_de_uso}

Entregue:
- entendimento do problema
- premissas adotadas
- análise das 4 arquiteturas
- arquitetura principal escolhida
- estrutura de agentes
- ferramentas sugeridas para cada agente
- fluxo operacional
- observações de execução
"""
    resultado = agente_orquestrador.run_sync(prompt)
    return resultado.output


def gerar_codigo_orquestrador(caso_de_uso: str, plano: PlanoArquiteturaMultiAgentes) -> CodigoOrquestrador:
    plano_json = plano.model_dump_json(indent=2, ensure_ascii=False)

    prompt = f"""
Gere o código do orquestrador com base no caso de uso e no plano abaixo.

CASO DE USO:
{caso_de_uso}

PLANO DA ARQUITETURA:
{plano_json}

Requisitos do código:
- Python completo
- Estrutura clara
- Código do orquestrador alinhado com a arquitetura principal escolhida
- Inclua agentes, schemas e fluxo de execução
- O código deve ser fácil de adaptar para um projeto real
- NÃO inclua sugestões de ferramentas
- NÃO use classes tradicionais como agentes
- Use apenas pydantic_ai.Agent para os agentes

Entregue:
- nome_arquivo
- codigo_python
- explicacao
- dependencias
- proximos_passos
"""
    resultado = agente_codigo.run_sync(prompt)
    return resultado.output


# =========================================================
# UTILITÁRIOS
# =========================================================

def plano_para_dict(plano: PlanoArquiteturaMultiAgentes) -> dict:
    return plano.model_dump()


def codigo_para_dict(codigo: CodigoOrquestrador) -> dict:
    return codigo.model_dump()


def render_tool(tool: ToolSugerida) -> None:
    st.markdown(
        f"""
**{tool.nome}**  
Tipo: `{tool.tipo}`  
Prioridade: **{tool.prioridade}**  
{tool.finalidade}
"""
    )


def render_plano(plano: PlanoArquiteturaMultiAgentes) -> None:
    st.subheader("Entendimento do problema")
    st.write(plano.entendimento_problema)

    st.subheader("Premissas")
    for item in plano.premissas:
        st.write(f"- {item}")

    st.subheader("Análise das arquiteturas")
    for a in plano.avaliacoes:
        with st.expander(f"{a.nome} — Adequação: {a.adequacao}", expanded=False):
            st.write("**Como funciona no contexto**")
            st.write(a.como_funciona_no_contexto)

            st.write("**Prós**")
            for p in a.pros:
                st.write(f"- {p}")

            st.write("**Contras**")
            for c in a.contras:
                st.write(f"- {c}")

    st.subheader("Arquitetura principal")
    st.success(plano.arquitetura_principal)

    st.subheader("Justificativa da escolha")
    st.write(plano.justificativa_escolha)

    if plano.arquitetura_secundaria:
        st.subheader("Arquitetura secundária")
        st.info(plano.arquitetura_secundaria)

    st.subheader("Estrutura de agentes")
    st.write(f"**Quantidade de agentes:** {plano.quantidade_de_agentes}")

    for ag in plano.agentes:
        with st.expander(f"{ag.nome}", expanded=False):
            st.write("**Responsabilidade**")
            st.write(ag.responsabilidade)

            st.write("**Ferramentas sugeridas**")
            for tool in ag.ferramentas_sugeridas:
                render_tool(tool)

    st.subheader("Fluxo operacional")
    for passo in sorted(plano.fluxo_operacional, key=lambda x: x.ordem):
        st.write(f"**{passo.ordem}.** {passo.descricao}")

    st.subheader("Observações de execução")
    for obs in plano.observacoes_execucao:
        st.write(f"- {obs}")


def render_codigo(codigo: CodigoOrquestrador) -> None:
    st.subheader("Arquivo sugerido")
    st.code(codigo.nome_arquivo, language="text")

    st.subheader("Explicação")
    st.write(codigo.explicacao)

    st.subheader("Dependências")
    for dep in codigo.dependencias:
        st.write(f"- {dep}")

    st.subheader("Código do orquestrador")
    st.code(codigo.codigo_python, language="python")

    st.subheader("Próximos passos")
    for passo in codigo.proximos_passos:
        st.write(f"- {passo}")


# =========================================================
# INTERFACE
# =========================================================

st.title("🧠 AI Architect para Multi-Agentes")
st.caption("Digite um caso de uso em linguagem natural e receba a arquitetura e o código do orquestrador.")

with st.sidebar:
    st.header("Como usar")
    st.write(
        """
1. Descreva o problema em linguagem natural.  
2. Clique em **Gerar arquitetura**.  
3. Depois clique em **Gerar código do orquestrador**.  
4. Veja o plano completo e baixe os arquivos.
"""
    )
    st.divider()
    st.write(f"**Modelo atual:** `{MODEL_NAME}`")
    st.write("**Saídas:** plano executivo + código Python")

exemplo_padrao = (
    "Quero criar um assistente que analise tendências de busca no Google Trends "
    "e use isso para sugerir territórios de marca com potencial de crescimento."
)

caso_de_uso = st.text_area(
    "Descreva o caso de uso em linguagem natural",
    value=exemplo_padrao,
    height=180,
    placeholder="Explique o problema, o objetivo e o contexto de negócio...",
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    gerar_plano = st.button("🚀 Gerar arquitetura", use_container_width=True)

with col2:
    gerar_codigo_btn = st.button("💻 Gerar código", use_container_width=True)

with col3:
    limpar = st.button("🧹 Limpar", use_container_width=True)

if limpar:
    st.session_state.pop("resultado_plano", None)
    st.session_state.pop("resultado_codigo", None)
    st.rerun()

if gerar_plano:
    if not caso_de_uso.strip():
        st.error("Você precisa descrever um caso de uso.")
    elif not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY não encontrada. Verifique seu arquivo .env.")
    else:
        with st.spinner("Analisando o caso de uso e projetando a arquitetura..."):
            try:
                plano = projetar_arquitetura(caso_de_uso)
                st.session_state["resultado_plano"] = plano
                st.success("Arquitetura gerada com sucesso.")
            except Exception as e:
                st.error(f"Erro ao gerar a arquitetura: {e}")

if gerar_codigo_btn:
    if not caso_de_uso.strip():
        st.error("Você precisa descrever um caso de uso.")
    elif not OPENROUTER_API_KEY:
        st.error("OPENROUTER_API_KEY não encontrada. Verifique seu arquivo .env.")
    else:
        if "resultado_plano" not in st.session_state or st.session_state["resultado_plano"] is None:
            with st.spinner("Primeiro gerando a arquitetura para depois criar o código..."):
                try:
                    plano = projetar_arquitetura(caso_de_uso)
                    st.session_state["resultado_plano"] = plano
                except Exception as e:
                    st.error(f"Erro ao gerar a arquitetura: {e}")
                    st.stop()

        with st.spinner("Gerando o código do orquestrador com base na recomendação final..."):
            try:
                codigo = gerar_codigo_orquestrador(caso_de_uso, st.session_state["resultado_plano"])
                st.session_state["resultado_codigo"] = codigo
                st.success("Código gerado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao gerar o código: {e}")

# =========================================================
# RESULTADOS
# =========================================================

if "resultado_plano" in st.session_state and st.session_state["resultado_plano"] is not None:
    st.divider()
    st.header("Plano de arquitetura")
    render_plano(st.session_state["resultado_plano"])

    st.divider()
    st.subheader("Download do plano")
    plano_json = json.dumps(
        plano_para_dict(st.session_state["resultado_plano"]),
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        label="Baixar plano em JSON",
        data=plano_json,
        file_name="plano_arquitetura_multi_agentes.json",
        mime="application/json",
        use_container_width=True,
    )

if "resultado_codigo" in st.session_state and st.session_state["resultado_codigo"] is not None:
    st.divider()
    st.header("Código do orquestrador")
    render_codigo(st.session_state["resultado_codigo"])

    st.divider()
    st.subheader("Download do código")
    st.download_button(
        label="Baixar código Python",
        data=st.session_state["resultado_codigo"].codigo_python,
        file_name=st.session_state["resultado_codigo"].nome_arquivo,
        mime="text/x-python",
        use_container_width=True,
    )

    st.subheader("Download do pacote completo em JSON")
    codigo_json = json.dumps(
        codigo_para_dict(st.session_state["resultado_codigo"]),
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        label="Baixar código e metadados em JSON",
        data=codigo_json,
        file_name="codigo_orquestrador.json",
        mime="application/json",
        use_container_width=True,
    )