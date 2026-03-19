from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

load_dotenv()


# =========================
# CONFIGURAÇÃO DO MODELO
# =========================

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
modelo = OpenRouterModel(MODEL_NAME)


# =========================
# SCHEMAS DE SAÍDA
# =========================

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


# =========================
# AGENTE ORQUESTRADOR
# =========================

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
- Ao sugerir ferramentas, pense em ferramentas reais ou classes de ferramentas que seriam úteis para cada agente
""",
)


# =========================
# FUNÇÃO PRINCIPAL
# =========================

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


# =========================
# UTILITÁRIOS DE EXIBIÇÃO
# =========================

def imprimir_plano(plano: PlanoArquiteturaMultiAgentes) -> None:
    print("\n" + "=" * 80)
    print("ENTENDIMENTO DO PROBLEMA")
    print("=" * 80)
    print(plano.entendimento_problema)

    print("\n" + "=" * 80)
    print("PREMISSAS")
    print("=" * 80)
    for p in plano.premissas:
        print(f"- {p}")

    print("\n" + "=" * 80)
    print("ANÁLISE DAS ARQUITETURAS")
    print("=" * 80)
    for a in plano.avaliacoes:
        print(f"\n[{a.nome}]")
        print(f"Adequação: {a.adequacao}")
        print(f"Como funciona: {a.como_funciona_no_contexto}")
        print("Prós:")
        for item in a.pros:
            print(f"  - {item}")
        print("Contras:")
        for item in a.contras:
            print(f"  - {item}")

    print("\n" + "=" * 80)
    print("ARQUITETURA PRINCIPAL")
    print("=" * 80)
    print(plano.arquitetura_principal)

    print("\n" + "=" * 80)
    print("JUSTIFICATIVA")
    print("=" * 80)
    print(plano.justificativa_escolha)

    if plano.arquitetura_secundaria:
        print("\n" + "=" * 80)
        print("ARQUITETURA SECUNDÁRIA")
        print("=" * 80)
        print(plano.arquitetura_secundaria)

    print("\n" + "=" * 80)
    print("ESTRUTURA DE AGENTES")
    print("=" * 80)
    print(f"Quantidade de agentes: {plano.quantidade_de_agentes}")
    for ag in plano.agentes:
        print(f"\n- {ag.nome}")
        print(f"  Responsabilidade: {ag.responsabilidade}")
        print("  Ferramentas sugeridas:")
        for t in ag.ferramentas_sugeridas:
            print(f"    • {t.nome} [{t.tipo} | {t.prioridade}]")
            print(f"      Finalidade: {t.finalidade}")

    print("\n" + "=" * 80)
    print("FLUXO OPERACIONAL")
    print("=" * 80)
    for passo in plano.fluxo_operacional:
        print(f"{passo.ordem}. {passo.descricao}")

    print("\n" + "=" * 80)
    print("OBSERVAÇÕES DE EXECUÇÃO")
    print("=" * 80)
    for obs in plano.observacoes_execucao:
        print(f"- {obs}")


# =========================
# EXEMPLO DE EXECUÇÃO
# =========================

if __name__ == "__main__":
    print("Descreva o caso de uso em linguagem natural:")
    caso = input(">>> ").strip()

    if not caso:
        raise ValueError("Você precisa informar um caso de uso.")

    plano = projetar_arquitetura(caso)
    imprimir_plano(plano)