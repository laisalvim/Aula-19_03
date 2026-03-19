from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY não encontrado no ambiente.")

modelo = OpenRouterModel(
    MODEL_NAME,
    provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
)

Arquitetura = Literal[
    "prompt_chaining",
    "parallelization",
    "routing",
    "orchestrator_workers",
]

# =========================================================
# SCHEMAS
# =========================================================

class AgentePlano(BaseModel):
    nome: str
    papel: str
    entrada: str
    saida: str


class DecisaoArquitetura(BaseModel):
    arquitetura: Arquitetura
    motivo: str
    estilo_execucao: Literal["sync", "async"]
    agentes: list[AgentePlano]
    fluxo: list[str]


class CodigoGerado(BaseModel):
    nome_arquivo: str
    codigo_python: str
    explicacao: str


# =========================================================
# AGENTE 1: AVALIADOR / DECISOR
# =========================================================

avaliador = Agent(
    model=modelo,
    output_type=DecisaoArquitetura,
    system_prompt="""
Você é um arquiteto de software especialista em sistemas multi-agentes.

Sua tarefa:
- analisar o caso de uso
- escolher UMA arquitetura
- definir agentes simples e funcionais
- descrever um fluxo curto e executável

Regras:
- escolha apenas uma entre: prompt_chaining, parallelization, routing, orchestrator_workers
- prefira a solução mais simples que funcione
- use no máximo 4 agentes
- cada agente deve ter responsabilidade única
- defina se a execução deve ser sync ou async
- não escreva código
- não seja abstrato demais
""",
)

# =========================================================
# UTILITÁRIOS
# =========================================================

def limpar_codigo(texto: str) -> str:
    texto = texto.strip()
    texto = re.sub(r"^```(?:python)?\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s*```$", "", texto)
    return texto.strip()


def validar_codigo(codigo: str) -> tuple[bool, str]:
    try:
        ast.parse(codigo)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} (linha {e.lineno}, coluna {e.offset})"


# =========================================================
# TEMPLATES DE CÓDIGO
# =========================================================

def template_prompt_chaining() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import os
        from dotenv import load_dotenv
        from pydantic import BaseModel
        from pydantic_ai import Agent
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        load_dotenv()

        MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY não encontrado.")

        modelo = OpenRouterModel(
            MODEL_NAME,
            provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
        )

        class Resumo(BaseModel):
            texto: str

        class Analise(BaseModel):
            texto: str

        class SaidaFinal(BaseModel):
            resposta: str

        agente_resumo = Agent(
            model=modelo,
            output_type=Resumo,
            system_prompt="Resuma a demanda em pontos claros e curtos.",
        )

        agente_analise = Agent(
            model=modelo,
            output_type=Analise,
            system_prompt="Analise o resumo e proponha a melhor direção.",
        )

        agente_final = Agent(
            model=modelo,
            output_type=SaidaFinal,
            system_prompt="Gere a resposta final simples e objetiva.",
        )

        def pipeline(caso_de_uso: str) -> SaidaFinal:
            r1 = agente_resumo.run_sync(caso_de_uso).output
            r2 = agente_analise.run_sync(r1.texto).output
            r3 = agente_final.run_sync(f"Resumo: {r1.texto}\\n\\nAnalise: {r2.texto}").output
            return r3

        if __name__ == "__main__":
            resultado = pipeline("Crie um assistente que analise pedidos de negócio e gere um plano simples.")
            print(resultado.resposta)
        """
    ).strip()


def template_parallelization() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import asyncio
        import os
        from dotenv import load_dotenv
        from pydantic import BaseModel
        from pydantic_ai import Agent
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        load_dotenv()

        MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY não encontrado.")

        modelo = OpenRouterModel(
            MODEL_NAME,
            provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
        )

        class ResumoTema(BaseModel):
            tema: str
            resumo: str

        class Sintese(BaseModel):
            texto: str

        agente_pesquisador = Agent(
            model=modelo,
            output_type=ResumoTema,
            system_prompt="Você resume o tema de forma clara e objetiva.",
        )

        agente_sintetizador = Agent(
            model=modelo,
            output_type=Sintese,
            system_prompt="Você sintetiza vários resumos em uma visão única e clara.",
        )

        async def pesquisar_tema(tema: str) -> ResumoTema:
            resultado = await agente_pesquisador.run(f"Resuma: {tema}")
            return resultado.output

        async def paralelo(temas: list[str]) -> Sintese:
            resultados = await asyncio.gather(*(pesquisar_tema(t) for t in temas))
            blocos = [
                f"TEMA: {r.tema}\\nRESUMO: {r.resumo}"
                for r in resultados
            ]
            sintese = await agente_sintetizador.run("\\n\\n".join(blocos))
            return sintese.output

        async def main() -> None:
            resultado = await paralelo([
                "machine learning",
                "deep learning",
                "processamento de linguagem natural",
            ])
            print(resultado.texto)

        if __name__ == "__main__":
            asyncio.run(main())
        """
    ).strip()


def template_routing() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import os
        from typing import Literal
        from dotenv import load_dotenv
        from pydantic import BaseModel
        from pydantic_ai import Agent
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        load_dotenv()

        MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY não encontrado.")

        modelo = OpenRouterModel(
            MODEL_NAME,
            provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
        )

        class Rota(BaseModel):
            destino: Literal["tecnologia", "geral"]
            justificativa: str

        class Resposta(BaseModel):
            resposta: str

        agente_roteador = Agent(
            model=modelo,
            output_type=Rota,
            system_prompt="Classifique a pergunta e escolha o melhor destino.",
        )

        agente_tecnologia = Agent(
            model=modelo,
            output_type=Resposta,
            system_prompt="Responda perguntas de tecnologia de forma clara e direta.",
        )

        agente_geral = Agent(
            model=modelo,
            output_type=Resposta,
            system_prompt="Responda perguntas gerais de forma clara e direta.",
        )

        def responder(pergunta: str) -> str:
            rota = agente_roteador.run_sync(pergunta).output

            if rota.destino == "tecnologia":
                return agente_tecnologia.run_sync(pergunta).output.resposta

            return agente_geral.run_sync(pergunta).output.resposta

        if __name__ == "__main__":
            print(responder("O que é machine learning?"))
        """
    ).strip()


def template_orchestrator_workers() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import os
        from dotenv import load_dotenv
        from pydantic import BaseModel
        from pydantic_ai import Agent
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        load_dotenv()

        MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY não encontrado.")

        modelo = OpenRouterModel(
            MODEL_NAME,
            provider=OpenRouterProvider(api_key=OPENROUTER_API_KEY),
        )

        class Resposta(BaseModel):
            texto: str

        agente_coletor = Agent(
            model=modelo,
            output_type=Resposta,
            system_prompt="Colete e organize a informação principal.",
        )

        agente_analista = Agent(
            model=modelo,
            output_type=Resposta,
            system_prompt="Analise o conteúdo e destaque os pontos importantes.",
        )

        agente_redator = Agent(
            model=modelo,
            output_type=Resposta,
            system_prompt="Transforme a análise em uma resposta final clara.",
        )

        def acionar_coletor(texto: str) -> str:
            return agente_coletor.run_sync(texto).output.texto

        def acionar_analista(texto: str) -> str:
            return agente_analista.run_sync(texto).output.texto

        def acionar_redator(texto: str) -> str:
            return agente_redator.run_sync(texto).output.texto

        def pipeline(caso_de_uso: str) -> str:
            base = acionar_coletor(caso_de_uso)
            analise = acionar_analista(base)
            saida = acionar_redator(f"Base: {base}\\n\\nAnalise: {analise}")
            return saida

        if __name__ == "__main__":
            print(pipeline("Crie um assistente multi-agentes simples para organizar demandas."))
        """
    ).strip()


def render_codigo(decisao: DecisaoArquitetura) -> CodigoGerado:
    if decisao.arquitetura == "prompt_chaining":
        codigo = template_prompt_chaining()
        nome = "prompt_chaining.py"
        explicacao = "Pipeline linear simples com três agentes em sequência."
    elif decisao.arquitetura == "parallelization":
        codigo = template_parallelization()
        nome = "parallelization.py"
        explicacao = "Execução paralela com síntese final."
    elif decisao.arquitetura == "routing":
        codigo = template_routing()
        nome = "routing.py"
        explicacao = "Roteador decide o especialista e encaminha a pergunta."
    else:
        codigo = template_orchestrator_workers()
        nome = "orchestrator_workers.py"
        explicacao = "Orquestrador chama workers simples por funções."
    return CodigoGerado(
        nome_arquivo=nome,
        codigo_python=codigo,
        explicacao=explicacao,
    )


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def avaliar(caso_de_uso: str) -> DecisaoArquitetura:
    prompt = f"""
Analise o caso de uso abaixo e devolva uma decisão simples e executável.

CASO DE USO:
{caso_de_uso}
"""
    return avaliador.run_sync(prompt).output


def gerar(caso_de_uso: str) -> tuple[DecisaoArquitetura, CodigoGerado]:
    decisao = avaliar(caso_de_uso)
    codigo = render_codigo(decisao)
    codigo.codigo_python = limpar_codigo(codigo.codigo_python)
    return decisao, codigo


def gerar_com_validacao(caso_de_uso: str) -> tuple[DecisaoArquitetura, CodigoGerado]:
    decisao, codigo = gerar(caso_de_uso)
    ok, erro = validar_codigo(codigo.codigo_python)

    if not ok:
        raise SyntaxError(f"Código inválido: {erro}")

    return decisao, codigo


# =========================================================
# EXEMPLO
# =========================================================

if __name__ == "__main__":
    caso = """
    Quero um assistente que analise uma demanda de negócio,
    escolha a arquitetura mais simples,
    e gere um código funcional de agentes.
    """

    decisao, codigo = gerar_com_validacao(caso)

    print("\n=== DECISÃO ===")
    print(decisao.model_dump_json(indent=2, ensure_ascii=False))

    print("\n=== ARQUIVO ===")
    print(codigo.nome_arquivo)

    print("\n=== EXPLICAÇÃO ===")
    print(codigo.explicacao)

    print("\n=== CÓDIGO ===")
    print(codigo.codigo_python)
