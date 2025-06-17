# Visão Computacional
> Durante a ação de aprendizado de maquina, existe a divisão de dados de treino e dados de teste, tal como:
> - Dados de treino são dados em que uma IA vai identificar e aprender padrões em imagens, a partir da intervenção humana, onde, o aprendizado torna-se supervisionado.
> - Dados de teste, consistem na parte onde será testado o “conhecimento” da máquina. Aqui, ela vai receber inumeros dados e identificar os padrões, desta vez, sem intervenção.

> Dessa forma, a Visão Computacional consiste em tornar uma máquina capaz de “enxergar”, identificar padrões (através dos pixels da imagem) e reconhece-las sem que haja interação humana,

---

# Heart Disease Dataset (Doença Cardíaca)
> Este dataset contém informações médicas de pacientes e é usado prever se um paciente tem ou não doença cardíaca com base em medições clínicas (frequência cardíaca, colesterol) e características pessoais (sexo, idade).

**Por que esse dataset é importante?**
- Representa um problema real de saúde pública, com variáveis que médicos realmente usam para avaliação.
- Serve tanto para exploração estatística quanto para machine learning supervisionado.
- Ajuda a ilustrar a aplicação da IA/visualização de dados em contextos médicos.

**O que dá pra fazer com ele?**
Com ferramentas como `matplotlib` ou `seaborn`, você pode:

- Comparar **frequência cardíaca** entre pacientes com e sem doença.
- Analisar o **efeito da idade** no risco cardíaco.
- Ver a **distribuição do colesterol** entre grupos.

Exemplos de perguntas que esse dataset pode responder:
- Pessoas mais velhas têm mais chance de doença cardíaca?
- Existe relação entre sexo e incidência de doença?
- Como a glicemia em jejum se relaciona com o risco cardíaco?

# Respondendo á pergunta: 
## Existe relação entre sexo e incidência de doença cardíaca?
```import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Dataset correto (Heart Disease UCI)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, header=None)

# Nomeando as colunas conforme a documentação original
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = columns

# Convertendo target para binário (0 = saudável, 1 = doença)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Mapeando sexo (1 = homem, 0 = mulher)
df['sex'] = df['sex'].map({1: 'Homem', 0: 'Mulher'})

# Plot CORRETO
plt.figure(figsize=(10,6))
ax = sns.countplot(x='sex', hue='target', data=df, palette='viridis')

# Configurações profissionais
plt.title('Distribuição de Doença Cardíaca por Gênero\n(Dados: UCI Cleveland)', 
          fontsize=14, pad=20)
plt.xlabel('Gênero', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.legend(['Saudável', 'Doença Cardíaca'], title='Status')

# Adicionando porcentagens
total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 3,
            f'{height}\n({height/total:.1%})',
            ha='center', 
            va='bottom',
            fontsize=10)

plt.tight_layout()
plt.show()
```

[Resultados da Análise](transferir.png)

A partir da análise deste conjunto de dados, 37.6% das mulheres possuem doença cardíaca, o que sugere uma possível associação entre sexo feminino e maior incidência de doenças cardíacas nesse contexto, sendo apenas uma análise de suposição, tendo em vista que o dataset possui dados de apenas 303 pessoas, muito pouco para uma análise com percentual afirmativo.
