import seaborn as sns
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