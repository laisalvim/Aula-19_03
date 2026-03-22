"""
╔══════════════════════════════════════════════════════════════════╗
║  Insights Comercial – Analisador de Causalidade de Demanda       ║
║  Off Trade | MVP v1.0                                            ║
╚══════════════════════════════════════════════════════════════════╝

Como rodar:
    pip install streamlit plotly pandas numpy requests
    streamlit run insights_comercial.py

LLM via OpenRouter (openai/gpt-4o-mini):
    - Defina sua chave na variável de ambiente OPENROUTER_API_KEY,
      ou insira diretamente no campo da sidebar do app.
    - Sem chave, o app exibe um insight simulado estático.
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────
import os
import requests
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insights Comercial | Causalidade de Demanda",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# ESTILOS CSS CORPORATIVOS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { padding: 1.2rem 2rem 2rem; max-width: 1400px; }

.page-header {
    background: linear-gradient(120deg, #0a1628 0%, #0f2240 60%, #0d1e38 100%);
    border: 1px solid #1b3358; border-radius: 10px;
    padding: 1.4rem 2rem; margin-bottom: 1.6rem;
    display: flex; align-items: center; justify-content: space-between;
}
.page-header h1 { font-size: 1.45rem; font-weight: 700; color: #ddeeff;
    letter-spacing: -.02em; margin: 0; }
.page-header p { font-size: .82rem; color: #5e8ab0; margin: .25rem 0 0; }
.pill { font-size:.65rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.1em; padding:3px 10px; border-radius:20px;
    background:#112a48; color:#4fa3d1; border:1px solid #1b3a60; }

.sec-title { font-size:.7rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.14em; color:#4fa3d1;
    border-bottom:1px solid #192b40; padding-bottom:.45rem;
    margin:1.4rem 0 .9rem; }

.kpi { background:#0d1a28; border:1px solid #1a2d42; border-radius:8px;
    padding:1rem 1.3rem; }
.kpi-label { font-size:.68rem; color:#5e7e9a; font-weight:500;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
.kpi-val { font-size:1.9rem; font-weight:700;
    font-family:'IBM Plex Mono',monospace; line-height:1; }
.kpi-sub { font-size:.76rem; margin-top:.28rem; font-weight:500; }
.pos{color:#34d399;} .neg{color:#f87171;} .neu{color:#94a3b8;}

.rank-wrap { display:flex; flex-direction:column; gap:.45rem; }
.rank-item { background:#0d1a28; border:1px solid #1a2d42;
    border-radius:7px; padding:.65rem 1rem; display:flex;
    align-items:center; gap:.75rem; }
.rank-num { font-size:.65rem; font-weight:700; color:#3d6a90;
    font-family:'IBM Plex Mono',monospace; width:1.2rem; text-align:right; }
.rank-name { font-size:.82rem; color:#c5d8e8; flex:1; }
.rank-bar-wrap { width:110px; height:6px; background:#1a2d42;
    border-radius:3px; overflow:hidden; }
.rank-bar { height:100%; border-radius:3px; }
.rank-pct { font-size:.78rem; font-family:'IBM Plex Mono',monospace;
    font-weight:600; width:4rem; text-align:right; }

.ftable { width:100%; border-collapse:collapse; font-size:.82rem; }
.ftable th { font-size:.66rem; font-weight:600; text-transform:uppercase;
    letter-spacing:.1em; color:#4a6a84; padding:.5rem .8rem;
    border-bottom:1px solid #192b40; text-align:left; }
.ftable td { padding:.55rem .8rem; border-bottom:1px solid #111e2c;
    color:#b8cede; vertical-align:middle; }
.ftable tr:hover td { background:#0f1e2e; }
.tag { font-size:.65rem; font-weight:600; padding:2px 7px;
    border-radius:4px; text-transform:uppercase; letter-spacing:.06em; }
.tag-int{background:#0d2a3a;color:#38bdf8;}
.tag-ext{background:#1a2a0d;color:#86efac;}
.tag-soc{background:#2a1a0d;color:#fbbf24;}
.tag-mac{background:#2a0d1a;color:#f9a8d4;}
.tag-comp{background:#1a0d2a;color:#c4b5fd;}
.conf-bar { display:inline-block; height:5px; border-radius:3px;
    background:#4fa3d1; vertical-align:middle; margin-right:5px; }

.insight-box { background:#090f1a; border:1px solid #1b3358;
    border-left:3px solid #4fa3d1; border-radius:8px;
    padding:1.4rem 1.6rem; line-height:1.8; color:#b8d0e8;
    font-size:.88rem; white-space:pre-wrap; }

section[data-testid="stSidebar"] { background:#08111d !important;
    border-right:1px solid #142030; }
section[data-testid="stSidebar"] .block-container { padding:.8rem 1rem; }
.stSelectbox label,.stDateInput label,.stTextInput label {
    font-size:.7rem !important; color:#4a6a84 !important;
    font-weight:500 !important; text-transform:uppercase !important;
    letter-spacing:.08em !important; }
.stButton > button { background:#0f2240; color:#c5d8e8;
    border:1px solid #1b3a60; border-radius:6px; font-weight:600;
    font-size:.8rem; padding:.5rem 1rem; width:100%;
    letter-spacing:.04em; transition:all .18s; }
.stButton > button:hover { background:#1b3a60; border-color:#4fa3d1; }
hr { border-color:#192b40 !important; margin:1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# DADOS DE REFERÊNCIA (FILTROS)
# ──────────────────────────────────────────────────────────────────
ESTADOS_CIDADES: dict[str, list[str]] = {
    "SP": ["São Paulo", "Campinas", "Ribeirão Preto", "Santos", "Sorocaba"],
    "RJ": ["Rio de Janeiro", "Niterói", "Petrópolis", "Volta Redonda"],
    "MG": ["Belo Horizonte", "Uberlândia", "Contagem", "Juiz de Fora"],
    "RS": ["Porto Alegre", "Caxias do Sul", "Pelotas", "Santa Maria"],
    "PR": ["Curitiba", "Londrina", "Maringá", "Foz do Iguaçu"],
    "BA": ["Salvador", "Feira de Santana", "Vitória da Conquista"],
    "PE": ["Recife", "Caruaru", "Petrolina"],
    "CE": ["Fortaleza", "Caucaia", "Juazeiro do Norte"],
}

CATALOGO: dict[str, dict[str, list[str]]] = {
    "Cerveja": {
        "BrandX Premium": ["Lata 350ml", "Long Neck 355ml", "Garrafa 600ml"],
        "BrandX Origin":  ["Lata 350ml", "Pack 6un", "Barril 5L"],
        "BrandZ Ultra":   ["Lata 269ml", "Lata 473ml"],
    },
    "Água Mineral": {
        "AquaPura":  ["500ml", "1,5L", "5L"],
        "FonteViva": ["350ml", "1L", "20L"],
    },
    "Refrigerante": {
        "FizzCola":    ["Lata 350ml", "PET 2L", "PET 600ml"],
        "CitrusBurst": ["Lata 350ml", "PET 1L"],
    },
    "Energético": {
        "TurboX":   ["250ml", "473ml"],
        "VitalPow": ["355ml", "710ml"],
    },
}

# ──────────────────────────────────────────────────────────────────
# SIMULAÇÃO DE DADOS
# ──────────────────────────────────────────────────────────────────

def _seed(estado: str, cidade: str, categoria: str, marca: str, sku: str) -> int:
    """Seed determinística a partir dos filtros, garante reprodutibilidade."""
    return abs(hash(f"{estado}{cidade}{categoria}{marca}{sku}")) % (2 ** 31)


def gerar_serie_temporal(data_inicio: date, data_fim: date, seed: int) -> pd.DataFrame:
    """Série temporal semanal: previsto vs realizado com dispersão simulada."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range(start=data_inicio, end=data_fim, freq="W")
    n = len(datas)
    if n == 0:
        return pd.DataFrame(columns=["semana", "previsto", "realizado", "dispersao_pct"])

    tendencia  = np.linspace(100, 108, n)
    sazonal    = 9 * np.sin(np.linspace(0, 4 * np.pi, n))
    previsto   = np.clip(tendencia + sazonal + rng.normal(0, 3, n), 50, 200)

    # Choques predominantemente negativos (ruptura, concorrência, clima)
    choque     = rng.uniform(-0.28, 0.06, n)
    realizado  = np.clip(previsto * (1 + choque) + rng.normal(0, 2, n), 30, 220)
    dispersao  = (realizado - previsto) / previsto * 100

    return pd.DataFrame({
        "semana":        datas,
        "previsto":      previsto.round(1),
        "realizado":     realizado.round(1),
        "dispersao_pct": dispersao.round(1),
    })


def gerar_fatores_causais(seed: int) -> pd.DataFrame:
    """Tabela de fatores com impacto simulado, ordenados por |impacto| desc."""
    rng = np.random.default_rng(seed)

    # (nome, tipo_legível, subtipo_tag_css)
    FATORES = [
        ("Ruptura de estoque",           "Interno",        "int"),
        ("Preço praticado vs tabelado",  "Interno",        "int"),
        ("Execução de trade / PDV",      "Interno",        "int"),
        ("Share de gôndola",             "Interno",        "int"),
        ("Elasticidade-preço",           "Interno",        "int"),
        ("Preço do concorrente",         "Concorrência",   "comp"),
        ("Promoção do concorrente",      "Concorrência",   "comp"),
        ("Gap de preço vs concorrente",  "Concorrência",   "comp"),
        ("Temperatura média",            "Externo",        "ext"),
        ("Volume de chuva",              "Externo",        "ext"),
        ("Eventos (feriados/jogos)",     "Externo",        "ext"),
        ("Google Trends",                "Externo",        "ext"),
        ("Buzz digital",                 "Externo",        "ext"),
        ("Renda média regional",         "Socioeconômico", "soc"),
        ("Densidade populacional",       "Socioeconômico", "soc"),
        ("Inflação (IPCA)",              "Macroeconômico", "mac"),
        ("Taxa de desemprego",           "Macroeconômico", "mac"),
    ]

    rows = []
    for nome, tipo, subtipo in FATORES:
        mag       = rng.uniform(1.0, 22.0)
        direcao   = rng.choice([-1, 1], p=[0.65, 0.35])
        impacto   = round(float(mag * direcao), 1)
        confianca = round(float(rng.uniform(55, 97)), 0)
        rows.append({
            "variavel":    nome,
            "tipo":        tipo,
            "subtipo":     subtipo,
            "impacto_pct": impacto,
            "direcao":     "↑" if impacto > 0 else "↓",
            "confianca":   confianca,
        })

    df = pd.DataFrame(rows)
    df = df.reindex(df["impacto_pct"].abs().sort_values(ascending=False).index)
    return df.reset_index(drop=True)


def resumo_kpis(df: pd.DataFrame) -> dict:
    """KPIs agregados do período."""
    return {
        "total_previsto":  round(df["previsto"].sum(), 0),
        "total_realizado": round(df["realizado"].sum(), 0),
        "dispersao_media": round(df["dispersao_pct"].mean(), 1),
        "semanas_abaixo":  int((df["dispersao_pct"] < 0).sum()),
        "total_semanas":   len(df),
    }

# ──────────────────────────────────────────────────────────────────
# GRÁFICO PRINCIPAL (Plotly)
# ──────────────────────────────────────────────────────────────────

def build_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Área sombreada entre as curvas
    fig.add_trace(go.Scatter(
        x=list(df["semana"]) + list(df["semana"][::-1]),
        y=list(df["previsto"]) + list(df["realizado"][::-1]),
        fill="toself", fillcolor="rgba(79,163,209,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=df["semana"], y=df["previsto"], mode="lines",
        name="Previsto",
        line=dict(color="#4fa3d1", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["semana"], y=df["realizado"], mode="lines+markers",
        name="Realizado",
        line=dict(color="#f97316", width=2.2),
        marker=dict(size=5, color="#f97316"),
    ))

    cores_disp = ["#34d399" if v >= 0 else "#f87171" for v in df["dispersao_pct"]]
    fig.add_trace(go.Bar(
        x=df["semana"], y=df["dispersao_pct"],
        name="Dispersão (%)", marker_color=cores_disp,
        opacity=0.5, yaxis="y2",
    ))

    fig.update_layout(
        paper_bgcolor="#0a1220", plot_bgcolor="#0a1220",
        font=dict(family="IBM Plex Sans", color="#8aacbf", size=11),
        legend=dict(orientation="h", y=1.08, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor="#131f2e", showgrid=True,
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Volume (índice)", gridcolor="#131f2e",
                   showgrid=True, zeroline=False,
                   tickfont=dict(size=10), titlefont=dict(size=10)),
        yaxis2=dict(title="Dispersão (%)", overlaying="y", side="right",
                    showgrid=False, zeroline=True,
                    zerolinecolor="#1e3a5f", zerolinewidth=1,
                    tickfont=dict(size=10), titlefont=dict(size=10),
                    tickformat="+.1f"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=340, hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1b2a", bordercolor="#1e3a5f",
                        font=dict(color="#c5d8e8", size=11)),
    )
    return fig

# ──────────────────────────────────────────────────────────────────
# HELPERS DE RENDERIZAÇÃO HTML
# ──────────────────────────────────────────────────────────────────
TAG_CLS = {"int":"tag-int","ext":"tag-ext","soc":"tag-soc","mac":"tag-mac","comp":"tag-comp"}
TAG_LBL = {"int":"Interno","ext":"Externo","soc":"Socioecon.","mac":"Macro","comp":"Concorrência"}


def render_kpi(label: str, value: str, sub: str, cls: str) -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-val {cls}">{value}</div>'
            f'<div class="kpi-sub {cls}">{sub}</div></div>')


def render_ranking(df: pd.DataFrame, top_n: int = 5) -> str:
    top = df.head(top_n)
    mx  = top["impacto_pct"].abs().max() or 1
    html = '<div class="rank-wrap">'
    for i, row in enumerate(top.itertuples(), 1):
        pct   = row.impacto_pct
        w     = int(abs(pct) / mx * 100)
        color = "#34d399" if pct >= 0 else "#f87171"
        sinal = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        cls   = "pos" if pct >= 0 else "neg"
        html += (
            f'<div class="rank-item">'
            f'<span class="rank-num">{i}</span>'
            f'<span class="rank-name">{row.variavel}</span>'
            f'<div class="rank-bar-wrap"><div class="rank-bar" '
            f'style="width:{w}%;background:{color}"></div></div>'
            f'<span class="rank-pct {cls}">{sinal}</span>'
            f'</div>'
        )
    return html + '</div>'


def render_tabela(df: pd.DataFrame) -> str:
    html = ('<table class="ftable"><thead><tr>'
            '<th>#</th><th>Variável</th><th>Tipo</th>'
            '<th>Impacto (%)</th><th>Dir.</th><th>Confiança</th>'
            '</tr></thead><tbody>')
    for i, row in enumerate(df.itertuples(), 1):
        tcls = TAG_CLS.get(row.subtipo, "tag-int")
        tlbl = TAG_LBL.get(row.subtipo, row.tipo)
        pct  = row.impacto_pct
        cls  = "pos" if pct >= 0 else "neg"
        sin  = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        bw   = int(row.confianca * 0.65)
        html += (
            f'<tr>'
            f'<td style="color:#3d6a90;font-family:\'IBM Plex Mono\',monospace">'
            f'{i:02d}</td>'
            f'<td>{row.variavel}</td>'
            f'<td><span class="tag {tcls}">{tlbl}</span></td>'
            f'<td class="{cls}" style="font-family:\'IBM Plex Mono\',monospace;'
            f'font-weight:600">{sin}</td>'
            f'<td style="color:#94a3b8">{row.direcao}</td>'
            f'<td><span class="conf-bar" style="width:{bw}px"></span>'
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:.76rem;'
            f'color:#6b84a0">{int(row.confianca)}%</span></td>'
            f'</tr>'
        )
    return html + '</tbody></table>'

# ──────────────────────────────────────────────────────────────────
# INTEGRAÇÃO LLM — OpenRouter (openai/gpt-4o-mini)
# ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Você é um analista de negócios especializado em explicar causalidade de demanda.\n"
    "Seu papel é interpretar dados e explicar, de forma clara e executiva, "
    "os fatores que levaram à diferença entre demanda prevista e realizada.\n"
    "Não invente dados. Não faça previsões. Não tome decisões operacionais."
)


def montar_prompt(filtros: dict, kpis: dict, top5: list[dict]) -> str:
    bullets = "\n".join(
        f"  {i+1}. {f['variavel']} ({f['tipo']}): "
        f"{'+'if f['impacto_pct']>=0 else ''}{f['impacto_pct']:.1f}% "
        f"[conf. {int(f['confianca'])}%]"
        for i, f in enumerate(top5)
    )
    return (
        f"Contexto:\n"
        f"- Produto: {filtros['sku']} | Marca: {filtros['marca']} | Categoria: {filtros['categoria']}\n"
        f"- Localidade: {filtros['cidade']} / {filtros['estado']}\n"
        f"- Período: {filtros['data_ini']} a {filtros['data_fim']}\n\n"
        f"KPIs:\n"
        f"- Volume previsto: {kpis['total_previsto']:.0f} | Realizado: {kpis['total_realizado']:.0f}\n"
        f"- Dispersão média: {kpis['dispersao_media']:+.1f}%\n"
        f"- Semanas abaixo do previsto: {kpis['semanas_abaixo']} de {kpis['total_semanas']}\n\n"
        f"Top 5 fatores (por impacto absoluto):\n{bullets}\n\n"
        f"Com base exclusivamente nesses dados, elabore um parágrafo executivo (4–7 linhas) "
        f"explicando o que ocorreu com a demanda e como os fatores se combinaram para "
        f"produzir a dispersão observada. Linguagem direta e sem jargão excessivo."
    )


def chamar_llm(api_key: str, prompt: str) -> str:
    """Chama OpenRouter e retorna o texto gerado."""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://insights-comercial.app",
                "X-Title": "Insights Comercial MVP",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as e:
        return f"[Erro HTTP {e.response.status_code}] {e.response.text[:300]}"
    except Exception as e:
        return f"[Erro ao chamar LLM] {e}"


def insight_fallback(kpis: dict, top5: list[dict]) -> str:
    """Insight estático quando não há chave de API."""
    disp   = kpis["dispersao_media"]
    label  = "abaixo" if disp < 0 else "acima"
    bullets = "\n".join(
        f"  • {f['variavel']}: {'+'if f['impacto_pct']>=0 else ''}{f['impacto_pct']:.1f}%"
        for f in top5[:3]
    )
    positivos = [f for f in top5[:5] if f["impacto_pct"] > 0]
    comp = ""
    if positivos:
        p = positivos[0]
        comp = (
            f"\n\n{p['variavel']} apresentou contribuição positiva de "
            f"+{p['impacto_pct']:.1f}%, porém não foi suficiente para "
            f"compensar os fatores de pressão negativa."
        )
    return (
        f"A demanda realizada ficou {abs(disp):.1f}% {label} do previsto "
        f"ao longo do período, com {kpis['semanas_abaixo']} de "
        f"{kpis['total_semanas']} semanas abaixo da meta.\n\n"
        f"Os principais fatores causais identificados foram:\n{bullets}\n\n"
        f"A combinação desses elementos criou uma pressão sistêmica sobre a demanda, "
        f"refletindo tanto aspectos operacionais internos quanto dinâmicas externas de mercado."
        f"{comp}"
    )

# ──────────────────────────────────────────────────────────────────
# SIDEBAR — FILTROS
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding:.6rem 0 1rem">
          <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.14em;color:#4fa3d1;margin-bottom:.25rem">Plataforma</div>
          <div style="font-size:1rem;font-weight:700;color:#ddeeff;line-height:1.2">
              Insights Comercial</div>
          <div style="font-size:.72rem;color:#3d6a90;margin-top:.1rem">
              Causalidade de Demanda · Off Trade</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sec-title" style="margin-top:0">Localidade</div>',
                unsafe_allow_html=True)
    estado_sel = st.selectbox("Estado", list(ESTADOS_CIDADES.keys()))
    cidade_sel = st.selectbox("Cidade", ESTADOS_CIDADES[estado_sel])

    st.markdown('<div class="sec-title">Produto</div>', unsafe_allow_html=True)
    cat_sel   = st.selectbox("Categoria", list(CATALOGO.keys()))
    marca_sel = st.selectbox("Marca", list(CATALOGO[cat_sel].keys()))
    sku_sel   = st.selectbox("SKU", CATALOGO[cat_sel][marca_sel])

    st.markdown('<div class="sec-title">Período</div>', unsafe_allow_html=True)
    hoje     = date.today()
    data_ini = st.date_input("Data início", value=hoje - timedelta(weeks=13),
                              max_value=hoje - timedelta(days=7))
    data_fim = st.date_input("Data fim", value=hoje,
                              min_value=data_ini + timedelta(days=7))

    st.divider()
    st.markdown('<div class="sec-title">LLM Config</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "OpenRouter API Key (opcional)", type="password",
        placeholder="sk-or-...",
        help="Sem chave, o insight é simulado localmente.",
    )
    api_key = api_key_input or os.environ.get("OPENROUTER_API_KEY", "")

    st.divider()
    analisar = st.button("🔍  Analisar", use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# HEADER DA PÁGINA
# ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div>
    <h1>Insights Comercial
      <span class="pill">MVP</span>
      <span class="pill">Off Trade</span>
    </h1>
    <p>Analisador de Causalidade de Demanda · Dados simulados · Diagnóstico por IA</p>
  </div>
  <div style="text-align:right">
    <div style="font-size:.7rem;color:#3d6a90;text-transform:uppercase;
        letter-spacing:.1em;margin-bottom:.2rem">Seleção ativa</div>
    <div style="font-size:.85rem;color:#c5d8e8;font-weight:600">
        {cat_sel} · {marca_sel} · {sku_sel}</div>
    <div style="font-size:.78rem;color:#4a6a84">
        {cidade_sel}/{estado_sel} &nbsp;·&nbsp;
        {data_ini.strftime('%d/%m/%Y')} – {data_fim.strftime('%d/%m/%Y')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ESTADO DE SESSÃO
# ──────────────────────────────────────────────────────────────────
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None

# ──────────────────────────────────────────────────────────────────
# DISPARO DA ANÁLISE
# ──────────────────────────────────────────────────────────────────
if analisar:
    seed = _seed(estado_sel, cidade_sel, cat_sel, marca_sel, sku_sel)

    with st.spinner("Simulando dados e calculando causalidade…"):
        df_serie = gerar_serie_temporal(data_ini, data_fim, seed)
        df_fat   = gerar_fatores_causais(seed)
        kpis     = resumo_kpis(df_serie)
        top5     = df_fat.head(5).to_dict("records")

    filtros = dict(
        estado=estado_sel, cidade=cidade_sel,
        categoria=cat_sel, marca=marca_sel, sku=sku_sel,
        data_ini=data_ini.strftime("%d/%m/%Y"),
        data_fim=data_fim.strftime("%d/%m/%Y"),
    )

    if api_key:
        with st.spinner("Gerando insight com IA (OpenRouter)…"):
            prompt      = montar_prompt(filtros, kpis, top5)
            insight_txt = chamar_llm(api_key, prompt)
    else:
        insight_txt = insight_fallback(kpis, top5)

    st.session_state["resultado"] = dict(
        df_serie=df_serie, df_fat=df_fat, kpis=kpis,
        top5=top5, insight=insight_txt, filtros=filtros,
    )

# ──────────────────────────────────────────────────────────────────
# RENDERIZAÇÃO DO DASHBOARD
# ──────────────────────────────────────────────────────────────────
res = st.session_state["resultado"]

if res is None:
    st.markdown("""
    <div style="margin-top:3rem;text-align:center;color:#3d6a90">
      <div style="font-size:2.5rem;margin-bottom:1rem">📊</div>
      <div style="font-size:1rem;font-weight:600;color:#5e8ab0;margin-bottom:.5rem">
          Configure os filtros e clique em <strong>Analisar</strong></div>
      <div style="font-size:.82rem;color:#2d4a60;max-width:400px;margin:auto">
          Selecione estado, cidade, categoria, marca, SKU e período
          na barra lateral para iniciar a análise de causalidade.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_serie = res["df_serie"]
df_fat   = res["df_fat"]
kpis     = res["kpis"]
top5     = res["top5"]
insight  = res["insight"]

# ── Seção 1 – KPIs ────────────────────────────────────────────────
st.markdown('<div class="sec-title">Resumo do Período</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
disp    = kpis["dispersao_media"]
cls_d   = "pos" if disp >= 0 else "neg"
txt_d   = f"{'acima' if disp >= 0 else 'abaixo'} do previsto"
pct_neg = round(kpis["semanas_abaixo"] / max(kpis["total_semanas"], 1) * 100, 0)

c1.markdown(render_kpi("Volume Previsto (índice)",
    f"{kpis['total_previsto']:,.0f}", "Acumulado no período", "neu"),
    unsafe_allow_html=True)
c2.markdown(render_kpi("Volume Realizado (índice)",
    f"{kpis['total_realizado']:,.0f}", "Acumulado no período",
    "pos" if kpis["total_realizado"] >= kpis["total_previsto"] else "neg"),
    unsafe_allow_html=True)
c3.markdown(render_kpi("Dispersão Média",
    f"{disp:+.1f}%", txt_d, cls_d), unsafe_allow_html=True)
c4.markdown(render_kpi("Semanas Abaixo do Previsto",
    str(kpis["semanas_abaixo"]),
    f"{pct_neg:.0f}% das {kpis['total_semanas']} semanas", "neg"),
    unsafe_allow_html=True)

# ── Seção 2 – Gráfico ─────────────────────────────────────────────
st.markdown('<div class="sec-title">Demanda Prevista vs Realizada</div>',
            unsafe_allow_html=True)
if len(df_serie) > 0:
    st.plotly_chart(build_chart(df_serie), use_container_width=True,
                    config={"displayModeBar": False})
else:
    st.warning("Intervalo de datas muito curto. Selecione ao menos 2 semanas.")

# ── Seção 3 – Ranking + Tabela ────────────────────────────────────
col_r, col_t = st.columns([1, 2], gap="large")
with col_r:
    st.markdown('<div class="sec-title">Top 5 Fatores de Impacto</div>',
                unsafe_allow_html=True)
    st.markdown(render_ranking(df_fat), unsafe_allow_html=True)
with col_t:
    st.markdown('<div class="sec-title">Todos os Fatores Causais</div>',
                unsafe_allow_html=True)
    st.markdown(render_tabela(df_fat), unsafe_allow_html=True)

# ── Seção 4 – Insight LLM ─────────────────────────────────────────
st.markdown('<div class="sec-title">Diagnóstico por IA</div>', unsafe_allow_html=True)
fonte = ("OpenRouter · openai/gpt-4o-mini" if api_key
         else "Insight simulado (adicione uma chave OpenRouter para IA real)")
st.markdown(
    f'<div style="font-size:.68rem;color:#3d6a90;margin-bottom:.6rem">'
    f'🤖 {fonte}</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

# ── Seção 5 – Dados Contextuais (expander) ────────────────────────
with st.expander("📋  Ver dados contextuais simulados (internos e externos)"):
    rng2 = np.random.default_rng(
        abs(hash(res["filtros"]["sku"] + res["filtros"]["cidade"])) % (2 ** 31)
    )
    ctx = {
        "Sell-out médio semanal (un)": f"{rng2.integers(800, 3500):,}",
        "Ruptura média (%)":           f"{rng2.uniform(5, 35):.1f}%",
        "Preço praticado (R$)":        f"R$ {rng2.uniform(4.5, 18.9):.2f}",
        "Preço tabelado (R$)":         f"R$ {rng2.uniform(4.8, 19.5):.2f}",
        "Execução de trade (score)":   f"{rng2.uniform(0.45, 0.95):.2f}",
        "Share de gôndola (%)":        f"{rng2.uniform(8, 35):.1f}%",
        "Preço do concorrente (R$)":   f"R$ {rng2.uniform(3.9, 17.5):.2f}",
        "Promoção concorrente ativa":  rng2.choice(["Sim", "Não"]),
        "Temperatura média (°C)":      f"{rng2.uniform(18, 34):.1f}°C",
        "Volume de chuva (mm)":        f"{rng2.uniform(0, 120):.0f} mm",
        "Google Trends (índice 0–100)":f"{rng2.integers(30, 100)}",
        "Buzz digital (menções/sem)":  f"{rng2.integers(200, 8000):,}",
        "Renda média regional (R$)":   f"R$ {rng2.uniform(1400, 6500):,.0f}",
        "Inflação acumulada IPCA (%)": f"{rng2.uniform(3.2, 8.5):.2f}%",
        "Taxa de desemprego (%)":      f"{rng2.uniform(6, 14):.1f}%",
    }
    items = list(ctx.items())
    half  = len(items) // 2
    ca, cb = st.columns(2)
    for k, v in items[:half]:
        ca.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:.4rem 0;border-bottom:1px solid #131f2e;font-size:.82rem">'
            f'<span style="color:#5e7e9a">{k}</span>'
            f'<span style="color:#b8cede;font-family:\'IBM Plex Mono\',monospace">{v}</span>'
            f'</div>', unsafe_allow_html=True)
    for k, v in items[half:]:
        cb.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:.4rem 0;border-bottom:1px solid #131f2e;font-size:.82rem">'
            f'<span style="color:#5e7e9a">{k}</span>'
            f'<span style="color:#b8cede;font-family:\'IBM Plex Mono\',monospace">{v}</span>'
            f'</div>', unsafe_allow_html=True)

# ── Rodapé ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid #131f2e;
    font-size:.68rem;color:#2d4a60;text-align:center">
    Insights Comercial MVP · Todos os dados são 100% simulados ·
    Nenhuma informação real é processada
</div>
""", unsafe_allow_html=True)