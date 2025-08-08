import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from PIL import Image
from datetime import datetime
import re

# ===================== Utils =====================
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', '_', name.strip()) or "colaborador"

def plot_radar(series_dict, title="Desempenho por Categoria"):
    """
    series_dict: dict {label: pd.Series(index=categorias, values=medias)}
    Plota 1 ou 2 séries (atual e anterior) com anotações de valores na série 'Atual'.
    """
    labels = list(next(iter(series_dict.values())).index)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([2,4,6,8,10])
    ax.set_yticklabels(['2','4','6','8','10'])
    ax.set_title(title, y=1.08)
    ax.grid(True)

    for i, (lbl, ser) in enumerate(series_dict.items()):
        vals = list(ser.values)
        vals_c = vals + vals[:1]
        ax.plot(angles_c, vals_c, linewidth=2, label=lbl)
        ax.fill(angles_c, vals_c, alpha=0.15)

        # Anota apenas na série "Atual"
        if lbl.lower().startswith("atual"):
            for ang, v in zip(angles, vals):
                ax.annotate(f"{v:.1f}", xy=(ang, v), xytext=(0,8),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    return fig

def perguntas_padrao_por_nivel(nivel: str) -> dict:
    # Baseado no seu modelo + incremento de complexidade por nível
    base = {
        "Versatilidade": [
            "Ajuda os colegas nas realizações das tarefas, visando alcançar os objetivos da equipe.",
            "Aceitação para inovações e atividades novas; executa tarefas com ânimo."
        ],
        "Relacionamento": [
            "Sabe trabalhar em equipe?",
            "Bom relacionamento com colegas.",
            "Autocontrole de emoções e comportamentos no trabalho."
        ],
        "Olhar sistêmico": [
            "Considera as necessidades dos clientes como fundamentais.",
            "Percebe o impacto do seu trabalho para a sociedade."
        ],
        "Trabalho em Equipe": [
            "Busca diálogo e troca de opiniões antes de escalar problemas.",
            "Trabalha em grupo sem causar conflitos e estimula participação coletiva."
        ],
        "Responsabilidade": [
            "Cumpre prazos e busca atingir objetivos do trabalho.",
            "Alcança níveis de qualidade conforme padrão do laboratório."
        ],
        "Foco em Resultados": [
            "Entrega atividades solicitadas; apoia organização e gestão do Kanban.",
            "Direciona esforços para objetivos da equipe."
        ],
        "Organização": [
            "Define prioridades para gerir múltiplas tarefas.",
            "Pontualidade nos horários de trabalho.",
            "Usa o tempo de forma adequada (ex.: evita celular em horário de trabalho)."
        ],
        "Norma 17025": [
            "Executa atividades da qualidade sem necessidade de cobrança (condições ambientais, manutenção preventiva, checagens, etc.).",
            "Conhecimento da norma e do manual da qualidade."
        ],
        "Técnica": [
            "Conhecimento nos procedimentos de calibração e métodos.",
            "Conhecimento sobre incerteza de medição.",
            "Busca aprender e aperfeiçoar técnicas no tempo disponível."
        ],
    }

    if nivel == "Assistente":
        base["Norma 17025"] += ["Registra não conformidades e ações corretivas quando necessário."]
        base["Técnica"] += ["Opera instrumentos sob supervisão seguindo ITs e boas práticas."]
    elif nivel == "Analista":
        base["Trabalho em Equipe"] += ["Apoia colegas tecnicamente e contribui na padronização de procedimentos."]
        base["Norma 17025"] += ["Mantém evidências, revisa formulários e registros do sistema da qualidade."]
        base["Técnica"] += ["Define setup de ensaios e interpreta resultados com mínima supervisão."]
    elif nivel == "Especialista":
        base["Trabalho em Equipe"] += ["Lidera tecnicamente frentes de trabalho e treinamentos internos."]
        base["Norma 17025"] += ["Conduz auditorias internas e representa o laboratório em auditorias externas."]
        base["Técnica"] += ["Investiga causas-raiz, define métodos/validações e orienta melhorias de medição."]
    # Estagiário: mantém base (foco em fundamentos)

    return base

def perguntas_editaveis(nivel: str):
    st.write("### Personalize as perguntas de cada categoria")
    categorias = st.text_area(
        "Categorias (separe por vírgula):",
        "Versatilidade,Relacionamento,Olhar sistêmico,Trabalho em Equipe,Responsabilidade,Foco em Resultados,Organização,Norma 17025,Técnica",
        key=f"cats_{nivel}"
    )
    categorias = [c.strip() for c in categorias.split(",") if c.strip()]
    defaults = perguntas_padrao_por_nivel(nivel)
    perguntas_dict = {}
    for cat in categorias:
        default = "\n".join(defaults.get(cat, []))
        perguntas_cat = st.text_area(f"Perguntas para **{cat}** (1 por linha):", default, key=f"ta_{nivel}_{cat}")
        perguntas_dict[cat] = [q.strip() for q in perguntas_cat.split("\n") if q.strip()]
    return perguntas_dict

# ===================== App =====================
st.title("Avaliação de Desempenho do Colaborador")

# Nível do cargo
nivel = st.selectbox("Nível do cargo (altera perguntas/skills padrão)", ["Estagiário", "Assistente", "Analista", "Especialista"])

with st.expander("🔧 Configurar perguntas (editar, incluir ou remover)"):
    perguntas_cfg = perguntas_editaveis(nivel)

col1, col2 = st.columns(2)
with col1:
    colaborador = st.text_input("Nome do colaborador avaliado")
with col2:
    avaliador = st.text_input("Nome do avaliador (opcional)")

data_hoje = st.date_input("Data da avaliação", datetime.today())

st.markdown("""
**Responda de 1 a 10 conforme a legenda:**

| Nota | Interpretação  |
|:---:|-----------------|
| 1-2 | **Nunca**       |
| 3-4 | **Raramente**   |
| 5-6 | **Às vezes**    |
| 7-8 | **Frequentemente** |
| 9-10| **Sempre**      |
""")

# Upload da última avaliação para comparação
prev_file = st.file_uploader("📈 (Opcional) Envie o CSV da **última avaliação** para comparar evolução", type=["csv"])

# Coleta atual
notas, categorias, perguntas_list = [], [], []
obs_por_categoria = {}

st.header("Preencha a avaliação")
for categoria, qs in perguntas_cfg.items():
    st.subheader(categoria)
    for i, q in enumerate(qs):
        val = st.slider(q, 1, 10, 5, key=f"sl_{nivel}_{categoria}_{i}")
        notas.append(val)
        categorias.append(categoria)
        perguntas_list.append(q)
    obs_cat = st.text_area(f"Observações sobre {categoria} (opcional):", key=f"obs_{nivel}_{categoria}")
    if obs_cat.strip():
        obs_por_categoria[categoria] = obs_cat.strip()

# Campos finais
pontos_positivos = st.text_area("✅ Pontos positivos (opcional):")
oportunidades = st.text_area("🔧 Oportunidades de melhorias (opcional):")

# ----------- Geração -----------
def gerar_pdf(nome_colaborador, nome_avaliador, data_avaliacao, nivel, df, media_atual,
              obs_categorias, pontos_pos, oportunidades, radar_buf, media_ant=None, delta=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatório de Avaliação de Desempenho", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Colaborador: {nome_colaborador}", ln=True)
    pdf.cell(0, 8, f"Avaliador: {nome_avaliador}", ln=True)
    pdf.cell(0, 8, f"Nível do cargo: {nivel}", ln=True)
    pdf.cell(0, 8, f"Data da avaliação: {data_avaliacao.strftime('%d/%m/%Y')}", ln=True)
    pdf.cell(0, 8, f"Média final: {df['Nota'].mean():.2f}", ln=True)
    pdf.ln(4)

    # Médias atuais (e deltas se houver)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Média por Categoria:", ln=True)
    pdf.set_font("Arial", "", 11)
    for cat, val in media_atual.items():
        if media_ant is not None and cat in media_ant.index:
            d = val - media_ant[cat]
            pdf.cell(0, 7, f"{cat}: {val:.2f}  (Δ {d:+.2f})", ln=True)
        else:
            pdf.cell(0, 7, f"{cat}: {val:.2f}", ln=True)

    pdf.ln(4)
    # Radar
    img = Image.open(radar_buf)
    tmp_path = "radar_tmp.png"
    img.save(tmp_path)
    pdf.image(tmp_path, x=45, w=120)
    pdf.ln(6)

    # Observações por categoria
    if obs_categorias:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Observações por Categoria:", ln=True)
        pdf.set_font("Arial", "", 11)
        for cat, texto in obs_categorias.items():
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 6, f"{cat}:")
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 6, texto)
            pdf.ln(1)

    if (pontos_pos or "").strip():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Pontos positivos:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, pontos_pos.strip())
        pdf.ln(2)

    if (oportunidades or "").strip():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Oportunidades de melhorias:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, oportunidades.strip())
        pdf.ln(2)

    # Perguntas e notas
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Perguntas e Notas:", ln=True)
    pdf.set_font("Arial", "", 10)
    for _, row in df.iterrows():
        pdf.multi_cell(0, 6, f"[{row['Categoria']}] {row['Pergunta']} - Nota: {row['Nota']}")
    return pdf

if st.button("Gerar Relatório"):
    if not perguntas_list:
        st.warning("Configure ao menos uma pergunta para gerar o relatório.")
    else:
        df = pd.DataFrame({"Categoria": categorias, "Pergunta": perguntas_list, "Nota": notas})
        st.dataframe(df)

        # Médias atuais
        media_atual = df.groupby("Categoria")["Nota"].mean().reindex(sorted(set(categorias)))

        # Histórico (opcional)
        media_ant = None
        delta_df = None
        if prev_file is not None:
            try:
                df_prev = pd.read_csv(prev_file)
                if set(["Categoria","Pergunta","Nota"]).issubset(df_prev.columns):
                    media_ant = df_prev.groupby("Categoria")["Nota"].mean().reindex(media_atual.index).fillna(np.nan)
                    delta_df = pd.DataFrame({
                        "Categoria": media_atual.index,
                        "Média Anterior": media_ant.values,
                        "Média Atual": media_atual.values,
                        "Δ (Atual - Anterior)": media_atual.values - media_ant.values
                    })
                    st.subheader("Comparativo com última avaliação")
                    st.dataframe(delta_df)

                    # Radar comparativo
                    fig = plot_radar({"Atual": media_atual, "Anterior": media_ant}, "Radar comparativo (Atual x Anterior)")
                else:
                    st.warning("O CSV enviado não possui as colunas esperadas: Categoria, Pergunta, Nota. Use o CSV exportado pelo app.")
                    fig = plot_radar({"Atual": media_atual}, "Desempenho por Categoria")
            except Exception as e:
                st.warning(f"Não foi possível ler o CSV enviado: {e}")
                fig = plot_radar({"Atual": media_atual}, "Desempenho por Categoria")
        else:
            fig = plot_radar({"Atual": media_atual}, "Desempenho por Categoria")

        st.subheader("Gráfico Radar")
        st.pyplot(fig)

        st.subheader("Média por Categoria")
        st.bar_chart(media_atual)

        st.write(f"**Média final do colaborador:** {df['Nota'].mean():.2f}")

        # CSV atual
        csv = df.to_csv(index=False).encode()
        st.download_button("Download do Relatório (CSV)", csv, "relatorio.csv")

        # Salvar radar p/ PDF
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        buf.seek(0)

        # PDF
        pdf = gerar_pdf(
            colaborador, avaliador, data_hoje, nivel,
            df, media_atual, obs_por_categoria, pontos_positivos, oportunidades,
            buf, media_ant=media_ant
        )
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_buf = BytesIO(pdf_bytes)

        nome_colab = sanitize_filename(colaborador)
        data_str = data_hoje.strftime("%Y-%m-%d")
        filename = f"relatorio_avaliacao_{nome_colab}_{nivel}_{data_str}.pdf"
        st.download_button("Baixar Relatório em PDF", pdf_buf, filename, mime="application/pdf")
