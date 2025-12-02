# Calibração Otimizada de Modelos de Volatilidade

## 📋 Descrição do Projeto

Este projeto implementa um sistema completo de **calibração de modelos de precificação de opções** usando **Algoritmos Evolutivos**. Desenvolvido como projeto final de Engenharia de Software, combina conceitos de finanças quantitativas, otimização evolutiva e boas práticas de desenvolvimento Python.

### Objetivo

Encontrar o conjunto ótimo de parâmetros de um modelo de precificação (Black-Scholes ou Heston) que minimize a diferença entre preços teóricos e preços reais de mercado através de um Algoritmo Genético.

### Principais Características

- ✅ **Modelo Black-Scholes**: Calibração de volatilidade implícita
- ✅ **Modelo Heston**: Suporte para volatilidade estocástica (5 parâmetros)
- ✅ **Algoritmo Evolutivo**: Seleção por torneio, crossover aritmético, mutação gaussiana
- ✅ **Gerador de Dados Sintéticos**: Cria opções realistas para testes
- ✅ **Interpolação de Taxas**: Carrega e interpola curva de juros livre de risco
- ✅ **Testes Completos**: Suite de testes com pytest (>90% coverage)
- ✅ **Código Pythonico**: Type hints, docstrings, PEP 8

---

## 🏗️ Arquitetura do Projeto

```
Final_PY/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Carregamento e interpolação de taxas de juros
│   ├── pricing_models.py        # Modelos Black-Scholes e Heston
│   ├── synthetic_data.py        # Gerador de opções sintéticas
│   └── evolutionary_algo.py     # Algoritmo Evolutivo (GA)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py      # Testes do carregador de dados
│   ├── test_pricing_models.py   # Testes dos modelos de precificação
│   └── test_evolutionary_algo.py # Testes do algoritmo evolutivo
├── main.py                      # Script principal de execução
├── databasePy.csv              # Curva de juros (Risk-Free Rate)
├── requirements.txt            # Dependências
└── README.md                   # Este arquivo
```

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes)

### Passo 1: Clone ou baixe o projeto

```bash
cd Final_PY
```

### Passo 2: Crie um ambiente virtual (recomendado)

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Passo 3: Instale as dependências

```powershell
pip install -r requirements.txt
```

---

## 💻 Como Usar

### Execução Básica

Execute o script principal:

```powershell
python main.py
```

Você verá um menu interativo:

```
Escolha o modo de execução:
  [1] Calibração Black-Scholes (Rápido - Recomendado)
  [2] Calibração com Portfólio Realista
  [3] Calibração Heston (Lento - Avançado)
  [4] Executar Todos

Opção [1]:
```

### Opção 1: Calibração Black-Scholes (Recomendada)

- Gera 30 opções sintéticas
- Calibra volatilidade usando algoritmo evolutivo
- Gera gráficos de convergência
- **Tempo estimado**: ~10-20 segundos

Exemplo de saída:

```
================================================================================
 CALIBRAÇÃO BLACK-SCHOLES COM ALGORITMO EVOLUTIVO
================================================================================

1. Carregando dados de taxa de juros
----------------------------------------
✓ RiskFreeRateLoader(251 registros, 2022-10-20 a 2022-12-30)

2. Gerando opções sintéticas
----------------------------------------
✓ Geradas 30 opções
✓ Volatilidade verdadeira: 0.2500 (25.00%)

...

6. Resultados da Calibração
----------------------------------------

Parâmetro            Verdadeiro    Calibrado    Erro (%)
------------------------------------------------------------
Volatilidade           0.250000     0.249872         0.05%

Métricas de Erro:
  MSE:  0.000123
  RMSE: 0.011089
```

### Opção 2: Portfólio Realista

Demonstra calibração usando um portfólio estruturado com diferentes strikes e maturidades (1M, 3M, 6M).

### Opção 3: Heston (Avançado)

⚠️ **Atenção**: Calibração Heston usa simulações Monte Carlo e pode levar vários minutos!

---

## 🧪 Executando os Testes

Este projeto possui uma suite completa de testes unitários.

### Executar todos os testes:

```powershell
pytest
```

### Com cobertura de código:

```powershell
pytest --cov=src --cov-report=html
```

Isso gera um relatório HTML em `htmlcov/index.html`.

### Executar testes específicos:

```powershell
# Apenas testes do data_loader
pytest tests/test_data_loader.py

# Apenas testes do pricing_models
pytest tests/test_pricing_models.py

# Apenas testes do evolutionary_algo
pytest tests/test_evolutionary_algo.py
```

### Testes com saída verbosa:

```powershell
pytest -v
```

---

## 📊 Estrutura dos Dados

### Arquivo `databasePy.csv`

Contém a curva de juros livre de risco do Tesouro dos EUA.

**Formato:**

```csv
Date,"1 Mo","2 Mo","3 Mo",...,"30 Yr"
12/30/2022,4.12,4.41,4.42,...,3.97
```

- **Colunas**: Date, 1 Mo, 2 Mo, 3 Mo, 4 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr, 20 Yr, 30 Yr
- **Taxas**: Em porcentagem (4.12 = 4.12%)
- **Uso**: O sistema interpola automaticamente para qualquer data/maturidade

---

## 🔬 Detalhes Técnicos

### Modelo Black-Scholes

**Fórmula para Call:**

$$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

onde:

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

- **S**: Preço spot
- **K**: Strike
- **T**: Tempo até vencimento
- **r**: Taxa livre de risco
- **σ**: Volatilidade (parâmetro a calibrar)

### Algoritmo Evolutivo

**Configuração padrão:**

- **População**: 100 indivíduos
- **Gerações**: 50
- **Seleção**: Torneio (tamanho 5)
- **Crossover**: Aritmético (taxa 80%)
- **Mutação**: Gaussiana (taxa 15%)
- **Elitismo**: 2 melhores

**Função Fitness:**

$$\text{Fitness} = \frac{1}{1 + \text{MSE}}$$

onde MSE (Mean Squared Error) é:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(P_{\text{modelo}}^i - P_{\text{mercado}}^i)^2$$

### Interpolação de Taxas

O sistema usa **interpolação linear** para:

1. **Interpolação temporal**: Entre datas não existentes
2. **Interpolação de maturidade**: Entre vencimentos não tabelados
3. **Tratamento de dados faltantes**: Forward/backward fill

---

## 📈 Resultados Esperados

Para calibração Black-Scholes com 30 opções sintéticas:

- **Erro típico**: < 1% na volatilidade
- **MSE**: ~ 0.0001 - 0.001
- **Convergência**: ~20-30 gerações
- **Tempo**: 10-20 segundos

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso                        |
| ---------- | ------ | -------------------------- |
| NumPy      | ≥1.24  | Computação numérica        |
| Pandas     | ≥2.0   | Manipulação de dados       |
| SciPy      | ≥1.10  | Interpolação e estatística |
| Matplotlib | ≥3.7   | Visualização               |
| Pytest     | ≥7.4   | Testes unitários           |

---

## 📝 Boas Práticas Implementadas

### Engenharia de Software

- ✅ **Modularização**: Código separado em módulos lógicos
- ✅ **Type Hints**: Tipagem estática em todas as funções
- ✅ **Docstrings**: Documentação completa (Google Style)
- ✅ **PEP 8**: Código formatado segundo convenções Python
- ✅ **DRY**: Don't Repeat Yourself
- ✅ **SOLID**: Princípios de design orientado a objetos

### Testes

- ✅ **Cobertura**: >90% do código testado
- ✅ **Testes unitários**: Cada função testada isoladamente
- ✅ **Testes de integração**: Fluxo completo validado
- ✅ **Fixtures**: Reutilização de setup de testes
- ✅ **Parametrização**: Testes com múltiplos cenários

### Tratamento de Erros

- ✅ **Validação de entrada**: Parâmetros inválidos rejeitados
- ✅ **Exceções customizadas**: Erros informativos
- ✅ **Interpolação robusta**: Lida com dados faltantes
- ✅ **Bounds clipping**: Parâmetros mantidos em limites válidos

---

## 🎯 Casos de Uso

### 1. Pesquisa Acadêmica

- Comparar diferentes configurações de algoritmos evolutivos
- Estudar convergência em problemas de otimização financeira

### 2. Prática Profissional

- Calibrar modelos de volatilidade em tempo real
- Gerar superfícies de volatilidade implícita

### 3. Educação

- Demonstrar conceitos de precificação de opções
- Ensinar algoritmos evolutivos com aplicação prática

---

## 🔄 Extensões Futuras

Possíveis melhorias:

1. **Modelos adicionais**: SABR, Local Volatility
2. **Otimizadores alternativos**: PSO, Differential Evolution
3. **Paralelização**: Usar multiprocessing para acelerar
4. **Interface gráfica**: Dash/Streamlit para visualização interativa
5. **Dados reais**: Integração com APIs de mercado (yfinance, Bloomberg)
6. **Machine Learning**: Redes neurais para calibração rápida

---

## 👨‍💻 Autor

**Pedro** - Engenharia de Software

Projeto desenvolvido para disciplina final de Engenharia de Software, demonstrando:

- Arquitetura modular
- Testes completos
- Documentação profissional
- Aplicação de algoritmos evolutivos em finanças quantitativas

---

## 📄 Licença

Este projeto é para fins educacionais.

---

## 🆘 Troubleshooting

### Erro: "Arquivo CSV não encontrado"

**Solução**: Certifique-se de que `databasePy.csv` está no diretório raiz do projeto.

### Erro ao importar módulos

**Solução**: Verifique se o ambiente virtual está ativado e dependências instaladas:

```powershell
pip install -r requirements.txt
```

### Testes falhando

**Solução**: Execute pytest com verbose para ver detalhes:

```powershell
pytest -v
```

### Calibração muito lenta

**Solução**: Reduza população ou gerações no `EvolutionaryConfig`:

```python
config = EvolutionaryConfig(
    population_size=50,  # Reduzido de 100
    n_generations=30     # Reduzido de 50
)
```

---

## 📚 Referências

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Goldberg, D. E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"
- Hull, J. (2018). "Options, Futures, and Other Derivatives"

---

## ✨ Agradecimentos

Agradecimentos especiais à comunidade Python e aos desenvolvedores das bibliotecas NumPy, SciPy e Pandas que tornaram este projeto possível.

---

**Versão**: 1.0.0  
**Data**: Dezembro 2025  
**Status**: ✅ Completo e Funcional
