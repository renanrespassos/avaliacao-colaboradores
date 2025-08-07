import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from PIL import Image
from datetime import datetime
import re

# ---------- Utilidades ----------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', '_', name.strip()) or "colaborador"

def plot_radar(media_categoria: pd.Series):
    labels = list(media_categoria.index)
    values = list(media_categoria.values)
    num_vars = len(labels)

    # Ângulos
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    values_c = values + values[:1]
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_c, values_c, linewidth=2)
    ax.fill(angles_c, values_c, alpha=0.25)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Desempenho por Categoria', y=1.08)
    ax.grid(True)

    # Anotar valores nos vértices
    for ang, val, lbl in zip(angles, values, labels):
        ax.annotate(f"{val:.1f}",
                    xy=(ang, val),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    return fig

def perguntas_default_por_categoria(cat: str) -> str:
    if cat == "Versatilidade":
        return ("Ajuda os colegas nas realizações das tarefas, visando alcançar os objetivos da equipe.\n"
                "Tem grande aceitação para inovações e atividades novas, desempenhando sempre com ânimo todas as tarefas.")
    if cat == "Relacionamento":
        return ("Sabe trabalhar em equipe?\n"
                "Apresenta um bom relacionamento com os colegas de trabalho?\n"
                "Apresenta autocontrole de suas emoções e comportamentos no local de trabalho.")
    if cat == "Olhar sistêmico":
        return ("Considera as necessidades dos clientes como fundamentais.\n"
                "Percebe a importância do seu trabalho e o impacto do mesmo em relação com a sociedade.")
    if cat == "Trabalho em Equipe":
        return ("Busca o diálogo e a troca de opiniões no grupo para que todos encontrem juntos a melhor solução quando necessário? "
                "Procura ajuda dos colegas mais próximos antes de solicitar ajuda dos signatários/metrologistas.\n"
                "Consegue trabalhar em grupo sem causar conflitos e estimulando a participação coletiva?")
    if cat == "Responsabilidade":
        return ("Cumpre seus prazos e busca atingir seus objetivos ao desempenhar seu trabalho?\n"
                "Procura alcançar altos níveis de qualidade conforme o padrão estabelecido pelo laboratório?")
    if cat == "Foco em Resultados":
        return ("Consegue realizar todas as atividades que são solicitadas, auxilia com a organização e gestão do Kanban.\n"
                "Direciona seus esforços para atingir os objetivos da equipe?")
    if cat == "Organização":
        return ("Sabe definir prioridades para alocar seu tempo de forma a desempenhar várias tarefas ao mesmo tempo?\n"
                "Pontualidade nos horários de trabalho.\n"
                "Sabe usar seu tempo de forma adequada? Evita a utilização do celular no horário de trabalho.")
    if cat == "Norma 17025":
        return ("Realiza as atividades da qualidade para a manutenção do laboratório sem a necessidade de cobranças? "
                "(Condições Ambientais / Manutenção Preventiva / Checagens / Outros)\n"
                "Apresenta conhecimento da norma e do manual da qualidade? Conseguiria representar o laboratório na Auditoria?")
    if cat == "Técnica":
        return ("Apresenta conhecimento pleno nos procedimentos de calibração e nos métodos?\n"
                "Apresenta conhecimento avançado sobre incerteza de medição?\n"
                "Quando possui tempo disponível procura aprender e aperfeiçoar técnicas?")
    return ""

def perguntas_editaveis():
    st.write("### Personalize as perguntas de cada categoria")
    categorias = st.text_area(
        "Categorias (separe por vírgula):",
        "Versatilidade,Relacionamento,Olhar sistêmico,Trabalho em Equipe,Responsabilidade,Foco em Resultados,Organização,Norma 17025,Técnica"
    )
    categorias = [c.strip() for c in categorias.split(",") if c.strip()]
    perguntas_dict = {}
    for cat in categorias:
        default = perguntas_default_por_categoria(cat)
        perguntas_cat = st.text_area(f"Perguntas para **{cat}** (1 por linha):", default, key=f"ta_{cat}")
        perguntas_dict[cat] = [q.strip() for q in perguntas_cat.split("\n") if q.strip()]
    return perguntas_dict

# ---------- Interface ----------
st.title("Avaliação de Desempenho do Colaborador")

with st.expander("🔧 Configurar perguntas (editar, incluir ou remover)"):
    perguntas_cfg = perguntas_editaveis()

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

# Coleta das respostas
notas, categorias, perguntas_list = [], [], []
obs_por_categoria = {}

st.header("Preencha a avaliação")
for categoria, qs in perguntas_cfg.items():
    st.subheader(categoria)
    for i, q in enumerate(qs):
        val = st.slider(q, 1, 10, 5, key=f"sl_{categoria}_{i}")
        notas.append(val)
        categorias.append(categoria)
        perguntas_list.append(q)

    # Campo de observações por categoria
    obs_cat = st.text_area(f"Observações sobre {categoria} (opcional):", key=f"obs_{categoria}")
    if obs_cat.strip():
        obs_por_categoria[categoria] = obs_cat.strip()

# Campos finais (substituem “Observações finais”)
pontos_positivos = st.text_area("✅ Pontos positivos (opcional):")
oportunidades = st.text_area("🔧 Oportunidades de melhorias (opcional):")

# ---------- Geração do relatório ----------
def gerar_pdf(nome_colaborador, nome_avaliador, data_avaliacao, df, media_categoria,
              obs_categorias, pontos_pos, oportunidades, radar_buf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatório de Avaliação de Desempenho", ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Colaborador: {nome_colaborador}", ln=True)
    pdf.cell(0, 8, f"Avaliador: {nome_avaliador}", ln=True)
    pdf.cell(0, 8, f"Data da avaliação: {data_avaliacao.strftime('%d/%m/%Y')}", ln=True)
    pdf.cell(0, 8, f"Média final: {df['Nota'].mean():.2f}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Média por Categoria:", ln=True)
    pdf.set_font("Arial", "", 11)
    for cat, val in media_categoria.items():
        pdf.cell(0, 7, f"{cat}: {val:.2f}", ln=True)

    pdf.ln(4)
    # Radar
    img = Image.open(radar_buf)
    tmp_path = "radar_tmp.png"
    img.save(tmp_path)
    pdf.image(tmp_path, x=50, w=110)
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

    # Pontos positivos
    if (pontos_pos or "").strip():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Pontos positivos:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, pontos_pos.strip())
        pdf.ln(2)

    # Oportunidades
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

        media_categoria = df.groupby("Categoria")["Nota"].mean().reindex(sorted(set(categorias)))

        st.subheader("Gráfico Radar")
        fig = plot_radar(media_categoria)
        st.pyplot(fig)

        st.subheader("Média por Categoria")
        st.bar_chart(media_categoria)

        st.write(f"**Média final do colaborador:** {df['Nota'].mean():.2f}")

        # CSV
        csv = df.to_csv(index=False).encode()
        st.download_button("Download do Relatório (CSV)", csv, "relatorio.csv")

        # Salvar radar para PDF
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
        buf.seek(0)

        # PDF
        pdf = gerar_pdf(
            colaborador, avaliador, data_hoje, df, media_categoria,
            obs_por_categoria, pontos_positivos, oportunidades, buf
        )
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_buf = BytesIO(pdf_bytes)

        nome_colab = sanitize_filename(colaborador)
        data_str = data_hoje.strftime("%Y-%m-%d")
        filename = f"relatorio_avaliacao_{nome_colab}_{data_str}.pdf"
        st.download_button("Baixar Relatório em PDF", pdf_buf, filename, mime="application/pdf")
