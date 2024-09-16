# Salve este código em um arquivo chamado 'app.py'

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import base64

def obter_dados_acoes(codigos):
    try:
        dados = yf.download(codigos, period='1y')['Adj Close']
        return dados
    except Exception as e:
        st.error(f"Não foi possível obter os dados: {e}")
        return None

def simular_portfolios(dados, num_simulacoes):
    retornos_diarios = dados.pct_change().dropna()
    medias_retornos = retornos_diarios.mean()
    covariancia = retornos_diarios.cov()

    num_acoes = len(dados.columns)
    resultados = []

    for _ in range(num_simulacoes):
        pesos = np.random.dirichlet(np.ones(num_acoes), size=1)[0]
        retorno_esperado = np.dot(pesos, medias_retornos) * 252
        risco = np.sqrt(np.dot(pesos.T, np.dot(covariancia * 252, pesos)))
        sharpe = retorno_esperado / risco
        resultados.append({
            'Retorno': retorno_esperado,
            'Risco': risco,
            'Sharpe': sharpe,
            'Pesos': pesos
        })

    return pd.DataFrame(resultados)

def plotar_frontier(resultados, melhor_portfolio):
    fig, ax = plt.subplots(figsize=(10,6))
    scatter = ax.scatter(resultados['Risco'], resultados['Retorno'], c=resultados['Sharpe'], cmap='viridis', alpha=0.7)
    ax.scatter(melhor_portfolio['Risco'], melhor_portfolio['Retorno'], color='red', s=100, label='Melhor Portfólio')
    ax.set_xlabel('Risco (Desvio Padrão Anualizado)')
    ax.set_ylabel('Retorno Esperado Anualizado')
    ax.set_title('Fronteira Eficiente')
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Índice de Sharpe')

    # Adicionar anotação com informações
    ax.annotate(
        f"Sharpe: {melhor_portfolio['Sharpe']:.2f}\nRetorno: {melhor_portfolio['Retorno']:.2%}\nRisco: {melhor_portfolio['Risco']:.2%}",
        (melhor_portfolio['Risco'], melhor_portfolio['Retorno']),
        textcoords="offset points",
        xytext=(50,50),
        ha='left',
        fontsize=12,  # Aumentar o tamanho da fonte
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")
    )

    st.pyplot(fig)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def main():
    # Configurar a página com o nome e o ícone
    st.set_page_config(
        page_title="Investir com Elas",
        page_icon="💰",
        layout="wide"
    )

    # CSS para estilização e plano de fundo
    def adicionar_estilo():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{get_base64_of_bin_file('background.jpg')}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .transparente {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 20px;
                border-radius: 10px;
            }}
            h1, h2, h3, h4 {{
                color: #2E4053;
            }}
            p, li {{
                font-size:18px;
            }}
            strong {{
                font-weight: bold;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    adicionar_estilo()

    # Inicializar o estado da página
    if 'pagina' not in st.session_state:
        st.session_state['pagina'] = 'Página Inicial'

    # Adicionar logo do projeto
    st.sidebar.image("logo.png", use_column_width=True)

    # Menu de navegação
    menu = ["Página Inicial", "Simulador de Portfólio"]
    escolha = st.sidebar.selectbox("Navegação", menu, index=menu.index(st.session_state['pagina']))

    if escolha == "Página Inicial":
        st.session_state['pagina'] = 'Página Inicial'
        # Conteúdo da página inicial
        st.markdown('<div class="transparente">', unsafe_allow_html=True)
        st.title("Investir com Elas")
        st.header("Bem-vindo ao Investir com Elas")
        st.markdown("""
        Esta aplicação tem como objetivo auxiliar pessoas que desejam aprender sobre investimentos em ações e montar seus próprios portfólios.
        """, unsafe_allow_html=True)

        st.subheader("A Matemática por Trás do Simulador")

        st.markdown(r"""
        A simulação é baseada na **Teoria Moderna de Carteiras** desenvolvida por **Harry Markowitz**, que busca otimizar a distribuição de ativos para maximizar o retorno esperado dado um nível de risco aceitável.

        **Retorno Esperado do Portfólio ($E(R_p)$):**

        $$
        E(R_p) = \sum_{i=1}^{n} w_i \cdot E(R_i)
        $$

        Onde:

        - $w_i$ é o **peso** (proporção de investimento) do ativo $i$ no portfólio.
        - $E(R_i)$ é o **retorno esperado** do ativo $i$.

        **Interpretação:** O retorno esperado do portfólio é a média ponderada dos retornos esperados dos ativos que o compõem.

        **Risco do Portfólio (Desvio Padrão $\sigma_p$):**

        $$
        \sigma_p = \sqrt{ \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij} }
        $$

        Onde:

        - $\sigma_{ij}$ é a **covariância** entre os retornos dos ativos $i$ e $j$.
        - $\sigma_{ij} = \rho_{ij} \sigma_i \sigma_j$, onde $\rho_{ij}$ é a **correlação** entre os ativos $i$ e $j$.

        **Interpretação:** O risco do portfólio leva em consideração não apenas o risco individual de cada ativo, mas também como eles interagem entre si.

        **Índice de Sharpe ($S_p$):**

        $$
        S_p = \frac{E(R_p) - R_f}{\sigma_p}
        $$

        Onde:

        - $E(R_p)$ é o retorno esperado do portfólio.
        - $R_f$ é a **taxa livre de risco** (assumimos zero na simulação).
        - $\sigma_p$ é o risco (desvio padrão) do portfólio.

        **Interpretação:** O Índice de Sharpe mede a relação entre o retorno excedente do portfólio (acima da taxa livre de risco) e seu risco. Quanto maior o Índice de Sharpe, melhor é a relação risco-retorno.

        **Teoria de Markowitz:**

        A teoria de Markowitz propõe que, para um dado nível de retorno esperado, existe um portfólio com o menor risco possível. A combinação de todos esses portfólios eficientes forma a **Fronteira Eficiente**. Investidores racionais devem escolher portfólios na fronteira eficiente, otimizando o equilíbrio entre retorno e risco.

        **Como a Simulação Funciona:**

        - **Geração de Portfólios Aleatórios:** O simulador gera milhares de portfólios com diferentes combinações de pesos.
        - **Cálculo de Retorno e Risco:** Para cada portfólio, calculamos o retorno esperado e o risco.
        - **Cálculo do Índice de Sharpe:** Avaliamos a eficiência de cada portfólio utilizando o Índice de Sharpe.
        - **Seleção do Melhor Portfólio:** O portfólio com o maior Índice de Sharpe é considerado o mais eficiente em termos de risco e retorno.

        """, unsafe_allow_html=True)

        st.subheader("Videoaulas sobre Finanças e Investimentos")
        st.write("A seguir, disponibilizamos algumas videoaulas ministradas por especialistas em finanças:")
        
        # Espaço para adicionar links de vídeos
        # Você pode substituir os links abaixo pelos vídeos que desejar
        video_urls = [
            "https://youtu.be/By9r8WlMkMw",
            "https://youtu.be/W-KeVi52atI",
            "https://youtu.be/Loj3Rv7xD2I"
        ]

        for url in video_urls:
            st.video(url)

        st.subheader("Profissionais Influentes no Mercado Financeiro")
        
        # Informações de profissionais influentes
        profissionais = [
            {
                "nome": "Sylvia Mathews Burwell",
                "contribuicoes": "Primeira mulher a servir como Diretora do Escritório de Gestão e Orçamento dos EUA e Secretária de Saúde e Serviços Humanos.",
                "citacao": "\"A liderança é uma série de comportamentos, não um papel para heróis.\"",
                "imagem": "sylvia_burwell.jpg"
            },
            {
                "nome": "Abigail Johnson",
                "contribuicoes": "CEO da Fidelity Investments, uma das maiores empresas de serviços financeiros do mundo.",
                "citacao": "\"Não se trata apenas de ser uma mulher. Trata-se de ser capaz e estar disposta a assumir riscos.\"",
                "imagem": "abigail_johnson.jpg"
            },
            {
                "nome": "Muriel Siebert",
                "contribuicoes": "Primeira mulher a possuir uma posição na Bolsa de Valores de Nova York.",
                "citacao": "\"Nunca se conformem com menos do que vocês merecem.\"",
                "imagem": "muriel_siebert.jpg"
            },
            {
                "nome": "Mary Barra",
                "contribuicoes": "CEO da General Motors, liderando a empresa em inovação e tecnologia.",
                "citacao": "\"Não tenha medo de assumir o que você não conhece. Confie em si mesmo para aprender.\"",
                "imagem": "mary_barra.jpg"
            }
        ]

        for profissional in profissionais:
            st.markdown(f"### {profissional['nome']}")
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(profissional['imagem'], width=150)
            with cols[1]:
                st.markdown(f"**Contribuições**: {profissional['contribuicoes']}")
                st.markdown(f"**Citação**: *{profissional['citacao']}*")
            st.markdown("---")

        st.subheader("A Importância da Educação Financeira")
        st.markdown("""
        Promover a educação financeira é essencial para capacitar indivíduos a tomarem decisões informadas sobre seus investimentos. Um mercado financeiro mais inclusivo e diversificado beneficia a todos, trazendo diferentes perspectivas e promovendo a inovação.
        """)

        # Botão para ir ao simulador
        if st.button("Ir para o Simulador"):
            st.session_state['pagina'] = "Simulador de Portfólio"
            st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    elif escolha == "Simulador de Portfólio":
        st.session_state['pagina'] = 'Simulador de Portfólio'
        st.markdown('<div class="transparente">', unsafe_allow_html=True)
        st.header("Monte seu Portfólio de Ações")

        st.sidebar.subheader("Parâmetros de Entrada")

        # Carregar a lista de ações
        @st.cache_data
        def carregar_lista_acoes():
            df_acoes = pd.read_csv('tickers_b3.csv')
            df_acoes['option'] = df_acoes['code'] + ' - ' + df_acoes['company']
            return df_acoes

        df_acoes = carregar_lista_acoes()
        lista_acoes = df_acoes['option'].tolist()

        # Seleção de ações com multiselect
        codigos_selecionados = st.sidebar.multiselect(
            "Digite o código ou nome da empresa e selecione as ações:",
            options=lista_acoes
        )

        investimento_total_input = st.sidebar.number_input(
            "Investimento Total (R$):",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

        if st.sidebar.button("Executar Simulação"):
            if not codigos_selecionados:
                st.error("Por favor, selecione pelo menos uma ação.")
                return

            with st.spinner('Obtendo dados e realizando simulações...'):
                # Extrair os códigos das ações selecionadas
                codigos = [codigo.split(' - ')[0] for codigo in codigos_selecionados]

                investimento_total = float(investimento_total_input)
                num_acoes = len(codigos)
                num_simulacoes = max(1000, num_acoes * 500)

                dados = obter_dados_acoes(codigos)
                if dados is None or dados.empty:
                    st.error("Dados insuficientes para as ações fornecidas.")
                    return

                resultados = simular_portfolios(dados, num_simulacoes)
                melhor_portfolio = resultados.loc[resultados['Sharpe'].idxmax()]

                plotar_frontier(resultados, melhor_portfolio)

                pesos_percentuais = melhor_portfolio['Pesos'] * 100
                alocacao = dict(zip(dados.columns, pesos_percentuais.round(2)))

                # Calcular o valor a ser investido em cada ação
                valores_investimento = {acao: peso/100 * investimento_total for acao, peso in alocacao.items()}

                st.subheader("Melhor Portfólio Encontrado")
                st.markdown(f"**Retorno Esperado Anualizado:** {melhor_portfolio['Retorno']:.2%}")
                st.markdown(f"**Risco Anualizado:** {melhor_portfolio['Risco']:.2%}")
                st.markdown(f"**Índice de Sharpe:** {melhor_portfolio['Sharpe']:.2f}")
                st.markdown(f"**Número de Carteiras Simuladas:** {num_simulacoes}")

                st.subheader("Distribuição de Investimento")
                alocacao_df = pd.DataFrame({
                    'Ação': alocacao.keys(),
                    'Peso (%)': alocacao.values(),
                    'Valor a Investir (R$)': valores_investimento.values()
                })
                alocacao_df = alocacao_df.merge(df_acoes[['code', 'company']], left_on='Ação', right_on='code')
                alocacao_df = alocacao_df[['Ação', 'company', 'Peso (%)', 'Valor a Investir (R$)']]

                st.table(alocacao_df.style.format({'Peso (%)': '{:.2f}%', 'Valor a Investir (R$)': 'R$ {:.2f}'}))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Seleção inválida.")

if __name__ == "__main__":
    main()
