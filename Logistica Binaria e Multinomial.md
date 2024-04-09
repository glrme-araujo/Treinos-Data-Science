# Modelos Logisticos Binarios e Multinomiais
É uma distribuição de probabilidade de ocorrência de determinado evento (Binominal) ou eventos(Multinomiais).Também conhecida como Distribuição de Bernoulli.
Ambos modelos utilizam a função de máxima verossimilhança, O método busca valores para os parâmetros de maneira a maximizar a probabilidade dos dados amostrados, dado o modelo assumido (no caso, distribuição normal).
# Regressão Binária
É uma distribuição de probabilidade de ocorrência de determinado evento (Binominal).
Os valores podem estar entre Não-Evento(0) e Evento(1).
## Importando bibliotecas e dados


```python
import pandas as pd                     # manipulação de dado em formato de dataframe
import seaborn as sns                   # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt         # biblioteca de visualização de dados
import statsmodels.api as sm            # biblioteca de modelagem estatística
import numpy as np                      # biblioteca para operações matemáticas multidimensionais
import statsmodels.formula.api as smf   # biblioteca para modelagem glm
import scipy.stats as st
from scipy import stats
from statsmodels.iolib.summary2 import summary_col
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score,roc_curve,auc
from statsmodels.discrete.discrete_model import MNLogit   


dados = pd.read_csv("atrasado.csv")
```

## Análise Exploratória

#### Visualizar os Dados


```python
dados
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estudante</th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gabriela</td>
      <td>0</td>
      <td>12.5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patricia</td>
      <td>0</td>
      <td>13.3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gustavo</td>
      <td>0</td>
      <td>13.4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leticia</td>
      <td>0</td>
      <td>23.5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Luiz Ovidio</td>
      <td>0</td>
      <td>9.5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Sheila</td>
      <td>1</td>
      <td>24.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pedro</td>
      <td>1</td>
      <td>10.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Horacio</td>
      <td>1</td>
      <td>9.4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Leandro</td>
      <td>1</td>
      <td>14.2</td>
      <td>10</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Estela</td>
      <td>1</td>
      <td>1.0</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>


A base dados conta com observações de alunos que chegaram atrasados a aula, informando a distância(km) percorrida e a quantidade de semáforos pegos durante o trajeto.
#### Características das variáveis


```python
dados.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   estudante  100 non-null    object 
     1   atrasado   100 non-null    int64  
     2   dist       100 non-null    float64
     3   sem        100 non-null    int64  
    dtypes: float64(1), int64(2), object(1)
    memory usage: 3.2+ KB
    

#### Estatísticas univariadas


```python
dados.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.590000</td>
      <td>14.073000</td>
      <td>10.210000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.494311</td>
      <td>4.330112</td>
      <td>1.578229</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>13.350000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>16.125000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>18.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Contagem dos Valores


```python
print('0 = não atrasado')
print('1 = atrasado')
dados['atrasado'].value_counts() 
```

    0 = não atrasado
    1 = atrasado
    




    atrasado
    1    59
    0    41
    Name: count, dtype: int64



## Estimação de um modelo logístico binário


```python
modelo = smf.glm(formula = 'atrasado ~ dist + sem', 
                                        data=dados,
                                        family= sm.families.Binomial()).fit()
                 
```

### Parâmetros do modelo


```python
modelo.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>atrasado</td>     <th>  No. Observations:  </th>  <td>   100</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td>    97</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     2</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>Logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -50.466</td>
</tr>
<tr>
  <th>Date:</th>            <td>Tue, 09 Apr 2024</td> <th>  Deviance:          </th> <td>  100.93</td>
</tr>
<tr>
  <th>Time:</th>                <td>14:23:36</td>     <th>  Pearson chi2:      </th>  <td>  86.7</td> 
</tr>
<tr>
  <th>No. Iterations:</th>          <td>7</td>        <th>  Pseudo R-squ. (CS):</th>  <td>0.2913</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  -26.1665</td> <td>    8.442</td> <td>   -3.100</td> <td> 0.002</td> <td>  -42.712</td> <td>   -9.621</td>
</tr>
<tr>
  <th>dist</th>      <td>    0.1904</td> <td>    0.076</td> <td>    2.493</td> <td> 0.013</td> <td>    0.041</td> <td>    0.340</td>
</tr>
<tr>
  <th>sem</th>       <td>    2.3629</td> <td>    0.795</td> <td>    2.972</td> <td> 0.003</td> <td>    0.804</td> <td>    3.921</td>
</tr>
</table>


As informações geradas pelo modelo, que servem  de métricas para analisar os paramêtros e a eficiência global do modelo.
As mais relevantes para analisar o modelo são o P-valor de cada variável pretidora, e Log-Likelihood do modelo.
O P-valor (P>|Z|) obtido para cada variável ficou abaixo do de 0.05 ou 5%, isto mostra que os dois Betas são estatisticamente significantes para a contrusção do modelo.
O Log-Liklihood quanto mais alto melhor, ele é um parâmetro de comparação entre dois modelos.
O Qui-quadrado(Pearson Chi2) mostra a correlação entre as variáveis.
### Adicionando as predições inteiras e seus percentuais de probabilidade na base de dados



```python
dados['resultado'] = modelo.predict()
dados['resultado(%)'] = modelo.predict()*100
dados
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estudante</th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
      <th>resultado</th>
      <th>resultado(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gabriela</td>
      <td>0</td>
      <td>12.5</td>
      <td>7</td>
      <td>0.000712</td>
      <td>0.071202</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patricia</td>
      <td>0</td>
      <td>13.3</td>
      <td>10</td>
      <td>0.498561</td>
      <td>49.856131</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gustavo</td>
      <td>0</td>
      <td>13.4</td>
      <td>8</td>
      <td>0.008903</td>
      <td>0.890254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leticia</td>
      <td>0</td>
      <td>23.5</td>
      <td>7</td>
      <td>0.005751</td>
      <td>0.575127</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Luiz Ovidio</td>
      <td>0</td>
      <td>9.5</td>
      <td>8</td>
      <td>0.004257</td>
      <td>0.425694</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Sheila</td>
      <td>1</td>
      <td>24.0</td>
      <td>10</td>
      <td>0.884040</td>
      <td>88.404017</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pedro</td>
      <td>1</td>
      <td>10.0</td>
      <td>10</td>
      <td>0.346606</td>
      <td>34.660579</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Horacio</td>
      <td>1</td>
      <td>9.4</td>
      <td>10</td>
      <td>0.321210</td>
      <td>32.120980</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Leandro</td>
      <td>1</td>
      <td>14.2</td>
      <td>10</td>
      <td>0.541301</td>
      <td>54.130138</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Estela</td>
      <td>1</td>
      <td>1.0</td>
      <td>13</td>
      <td>0.991348</td>
      <td>99.134795</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 6 columns</p>
</div>


Agora cada observação conta com a presença da coluna de resultado, que foi predito utilizando o modelo desenvovlido.
Assim pode se ver a probabilidade de ocorrência para cada observação.
## Métricas para análise do modelo
Ao realizar uma Regressão Binária, é fundamental determinar um valor de cutt-off que irá considerar uma faixa para as probabilidades preditas serem eventos ou não-eventos. 
Os valores podem assumir números entre 0 e 1.
#### Construção de Função para matriz de confusão para determinado Cutoff


```python
def matriz_confusao(observado,predicts,cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
    
    for item in values:
        if item >= cutoff:
            predicao_binaria.append(1)
        else:
            predicao_binaria.append(0)
    
    cm = confusion_matrix(observado, predicao_binaria)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidadee],
                                'Acurácia':[acuracia]})
    return indicadores
```

#### Rodando Função de matriz de Confusão para Cuttof = 0.5


```python
matriz_confusao(observado=dados['atrasado'],
                predicts=dados['resultado'], 
                cutoff=0.5)
```


    
![png](output_29_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sensitividade</th>
      <th>Especificidade</th>
      <th>Acurácia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.779661</td>
      <td>0.609756</td>
      <td>0.71</td>
    </tr>
  </tbody>
</table>
</div>


Ao escolher o cut-off de 0.5, estamos considerando que as predições com valores maiores do que 0.5 (50% de probabilidade) serão considerados eventos, e as probabilidades menores do que esse cut-off será não-evento.
Então é calculada uma matriz de confusão para comparar as predições com os resultados reias.
A taxa de acerto dos eventos é chamada de sensitividade - foram acertos 46 de 62 possíveis não-eventos, 77%
A taxa de acerto dos não-eventos é chamada de especifidade - foram acertos 25 de 38 possíveis não-eventos, 60%
A acurácia mostra a taxa global de acertos do modelo para esse cut-off - Foram acertos no total 71/100 observações, 71% 
### Função para testar cutoffs entre 0,1 até 1.01


```python
def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado
```

### Rodando Função nos Dados 


```python
dados_plotagem = espec_sens(observado = dados['atrasado'],
                            predicts = dados['resultado'])
dados_plotagem

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cutoffs</th>
      <th>sensitividade</th>
      <th>especificidade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>1.000000</td>
      <td>0.170732</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>1.000000</td>
      <td>0.170732</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>1.000000</td>
      <td>0.170732</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>1.000000</td>
      <td>0.170732</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.96</td>
      <td>0.135593</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>0.135593</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>0.135593</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>0.101695</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 3 columns</p>
</div>



### Plotando Gráfico Sensitividade/Especificidade em função do Cuttof


```python
plt.figure(figsize=(10,10))
plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, '-o',
         color="#440154FF")
plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, '-o',
         color="#2ecc71")
plt.legend(['Sensitividade', 'Especificidade'], fontsize=17)
plt.xlabel("Cuttoff", fontsize=14)
plt.ylabel("Sensitividade / Especificidade", fontsize=14)
plt.show()
```


    
![png](output_36_0.png)
    

Esse grafico mostra as taxas de especifidade/ sensitividade variando em função do cutt-off.
### Plotagem Curva ROC


```python
fpr, tpr, thresholds = roc_curve(dados['atrasado'],dados['resultado'])
roc_auc = auc(fpr, tpr)

#Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

#Plotando a curva ROC
plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, '-o', color="#9b59b6")
plt.plot(fpr,fpr, color='gray')
plt.title("Área abaixo da curva: %f" % roc_auc +
          " | Coeficiente de GINI: %f" % gini, fontsize=17)
plt.xlabel("Especificidade", fontsize=14)
plt.ylabel("Sensitividade", fontsize=14)
plt.show()
```


    
![png](output_39_0.png)
    


A curva roc é um dos parâmetros que mais informa a qualidade global do modelo.
a curva roc é uma curva da especifidade em função da sensitividade, criando uma área de acerto que mostra demonstra a eficiência do modelo.
Quanto maior a área, melhor o ajuste do modelo.

# Regressão Multinomial
É uma distribuição de probabilidade de ocorrência de multiplos eventos.
Os valores podem estar entre referência(0), Evento-1(1), Evento-2(2), Evento-x(x).
É calculado a probabilidade de ocorrência de cada evento possível para determinada observação.
## Importando bibliotecas e dados


```python
import pandas as pd                     # manipulação de dado em formato de dataframe
import seaborn as sns                   # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt         # biblioteca de visualização de dados
import statsmodels.api as sm            # biblioteca de modelagem estatística
import numpy as np                      # biblioteca para operações matemáticas multidimensionais
import statsmodels.formula.api as smf   # biblioteca para modelagem glm
import scipy.stats as st
from statsmodels.iolib.summary2 import summary_col
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score,roc_curve,auc
from statsmodels.discrete.discrete_model import MNLogit   


dados_multinomial = pd.read_csv("atrasado_multinomial.csv",
                   delimiter = ',')
```

## Análise Exploratória

### Visualização dos dados


```python
dados_multinomial
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estudante</th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gabriela</td>
      <td>chegou atrasado segunda aula</td>
      <td>20.500000</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patricia</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.299999</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gustavo</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.400000</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leticia</td>
      <td>chegou atrasado segunda aula</td>
      <td>31.500000</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Luiz Ovidio</td>
      <td>chegou atrasado segunda aula</td>
      <td>17.500000</td>
      <td>16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Sheila</td>
      <td>nao chegou atrasado</td>
      <td>24.000000</td>
      <td>10</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pedro</td>
      <td>chegou atrasado primeira aula</td>
      <td>15.000000</td>
      <td>15</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Horacio</td>
      <td>chegou atrasado primeira aula</td>
      <td>14.400000</td>
      <td>15</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Leandro</td>
      <td>chegou atrasado segunda aula</td>
      <td>22.200001</td>
      <td>18</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Estela</td>
      <td>nao chegou atrasado</td>
      <td>1.000000</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>


A base dados conta com observações de alunos que chegaram atrasados para primeira aula, ou atrasados para segunda aula ou os que não chegaram atrasados, informando a distância(km) percorrida e a quantidade de semáforos pegos durante o trajeto.
### Características das variáveis


```python
dados_multinomial.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   estudante  100 non-null    object 
     1   atrasado   100 non-null    object 
     2   dist       100 non-null    float64
     3   sem        100 non-null    int64  
    dtypes: float64(1), int64(1), object(2)
    memory usage: 3.2+ KB
    

### Estatísticas das variáveis


```python
dados_multinomial.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dist</th>
      <th>sem</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>17.673000</td>
      <td>13.810000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.130299</td>
      <td>3.329376</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>14.950000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18.750000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>21.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>31.500000</td>
      <td>19.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Contagem dos valores


```python
dados_multinomial['atrasado'].value_counts(sort=False)
```




    atrasado
    chegou atrasado segunda aula     35
    chegou atrasado primeira aula    16
    nao chegou atrasado              49
    Name: count, dtype: int64



## Estimação do Modelo

### Criando Categorias de Referencia 


```python
dados_multinomial.loc[dados_multinomial['atrasado']==
                            'nao chegou atrasado',
                            'atrasado2'] = 0 #categoria de referência
dados_multinomial.loc[dados_multinomial['atrasado']==
                            'chegou atrasado primeira aula',
                            'atrasado2'] = 1
dados_multinomial.loc[dados_multinomial['atrasado']==
                            'chegou atrasado segunda aula',
                            'atrasado2'] = 2
```
É necessário definir um evento como referência antes de estimar o modelo.
E também criar valores para cada evento possivel, assim será possivel estimar o modelo corretamente.
Foi definido que
nao chegou atrasado = 0
chegou atrasado primeira aula = 1
chegou atrasado segunda aula = 2

```python
dados_multinomial
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estudante</th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
      <th>atrasado2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gabriela</td>
      <td>chegou atrasado segunda aula</td>
      <td>20.500000</td>
      <td>15</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patricia</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.299999</td>
      <td>18</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gustavo</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.400000</td>
      <td>16</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leticia</td>
      <td>chegou atrasado segunda aula</td>
      <td>31.500000</td>
      <td>15</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Luiz Ovidio</td>
      <td>chegou atrasado segunda aula</td>
      <td>17.500000</td>
      <td>16</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Sheila</td>
      <td>nao chegou atrasado</td>
      <td>24.000000</td>
      <td>10</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pedro</td>
      <td>chegou atrasado primeira aula</td>
      <td>15.000000</td>
      <td>15</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Horacio</td>
      <td>chegou atrasado primeira aula</td>
      <td>14.400000</td>
      <td>15</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Leandro</td>
      <td>chegou atrasado segunda aula</td>
      <td>22.200001</td>
      <td>18</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Estela</td>
      <td>nao chegou atrasado</td>
      <td>1.000000</td>
      <td>13</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



### Estimação do Modelo


```python
x = dados_multinomial.drop(columns=['estudante','atrasado','atrasado2'])
y = dados_multinomial['atrasado2']
X = sm.add_constant(x)
modelo_atrasado = MNLogit(endog=y, exog=X).fit()
```

    Optimization terminated successfully.
             Current function value: 0.245118
             Iterations 10
    
Estimação do modelo
Atrasado2 ~ Distancia e Semaforos
### Parâmetros do modelo


```python
modelo_atrasado.summary()
```




<table class="simpletable">
<caption>MNLogit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>atrasado2</td>    <th>  No. Observations:  </th>  <td>   100</td>  
</tr>
<tr>
  <th>Model:</th>                <td>MNLogit</td>     <th>  Df Residuals:      </th>  <td>    94</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 05 Apr 2024</td> <th>  Pseudo R-squ.:     </th>  <td>0.7574</td>  
</tr>
<tr>
  <th>Time:</th>                <td>16:10:13</td>     <th>  Log-Likelihood:    </th> <td> -24.512</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -101.02</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>4.598e-32</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>atrasado2=1</th>    <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>  -33.1352</td> <td>   12.183</td> <td>   -2.720</td> <td> 0.007</td> <td>  -57.014</td> <td>   -9.256</td>
</tr>
<tr>
  <th>dist</th>        <td>    0.5588</td> <td>    0.243</td> <td>    2.297</td> <td> 0.022</td> <td>    0.082</td> <td>    1.036</td>
</tr>
<tr>
  <th>sem</th>         <td>    1.6699</td> <td>    0.577</td> <td>    2.895</td> <td> 0.004</td> <td>    0.539</td> <td>    2.801</td>
</tr>
<tr>
  <th>atrasado2=2</th>    <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>  -62.2922</td> <td>   14.675</td> <td>   -4.245</td> <td> 0.000</td> <td>  -91.055</td> <td>  -33.530</td>
</tr>
<tr>
  <th>dist</th>        <td>    1.0784</td> <td>    0.302</td> <td>    3.566</td> <td> 0.000</td> <td>    0.486</td> <td>    1.671</td>
</tr>
<tr>
  <th>sem</th>         <td>    2.8949</td> <td>    0.686</td> <td>    4.220</td> <td> 0.000</td> <td>    1.550</td> <td>    4.239</td>
</tr>
</table>



Podemos interpertrar da mesma forma que o modelo binário. Só que agora temos duas equações para se analisar, ja que temos 2 possiveis eventos.
As mais relevantes para analisar o modelo são o P-valor de cada variável pretidora, e Log-Likelihood do modelo
Equação para o evento 1 e equação para evento 2.

### Qui²


```python
def Qui2(modelo_multinomial):
    maximo = modelo_multinomial.llf
    minimo = modelo_multinomial.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,1)
    df = pd.DataFrame({'Qui quadrado':[qui2],
                       'pvalue':[pvalue]})
    return df
Qui2(modelo_atrasado)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Qui quadrado</th>
      <th>pvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>153.014842</td>
      <td>3.802150e-35</td>
    </tr>
  </tbody>
</table>
</div>



## Predições

### Realizando Predição e adicionando ao banco de dados (Probabilidades)


```python
resultado = modelo_atrasado.predict()
resultado = pd.DataFrame(resultado)
dados_multinomial = pd.concat([dados_multinomial, resultado], axis=1)
```

### Adicionando maior probabilidade da predição


```python
classificacao = resultado.idxmax(axis=1)
dados_multinomial['predicao'] = classificacao
dados_multinomial
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estudante</th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
      <th>atrasado2</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>predicao</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gabriela</td>
      <td>chegou atrasado segunda aula</td>
      <td>20.500000</td>
      <td>15</td>
      <td>2.0</td>
      <td>1.801024e-02</td>
      <td>0.523388</td>
      <td>4.586018e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patricia</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.299999</td>
      <td>18</td>
      <td>2.0</td>
      <td>2.751301e-06</td>
      <td>0.018737</td>
      <td>9.812605e-01</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gustavo</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.400000</td>
      <td>16</td>
      <td>2.0</td>
      <td>6.796190e-04</td>
      <td>0.173472</td>
      <td>8.258489e-01</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leticia</td>
      <td>chegou atrasado segunda aula</td>
      <td>31.500000</td>
      <td>15</td>
      <td>2.0</td>
      <td>2.759476e-07</td>
      <td>0.003748</td>
      <td>9.962518e-01</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Luiz Ovidio</td>
      <td>chegou atrasado segunda aula</td>
      <td>17.500000</td>
      <td>16</td>
      <td>2.0</td>
      <td>2.083782e-02</td>
      <td>0.601588</td>
      <td>3.775739e-01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Sheila</td>
      <td>nao chegou atrasado</td>
      <td>24.000000</td>
      <td>10</td>
      <td>0.0</td>
      <td>9.531361e-01</td>
      <td>0.046317</td>
      <td>5.471598e-04</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pedro</td>
      <td>chegou atrasado primeira aula</td>
      <td>15.000000</td>
      <td>15</td>
      <td>1.0</td>
      <td>4.146183e-01</td>
      <td>0.557343</td>
      <td>2.803830e-02</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Horacio</td>
      <td>chegou atrasado primeira aula</td>
      <td>14.400000</td>
      <td>15</td>
      <td>1.0</td>
      <td>5.008256e-01</td>
      <td>0.481441</td>
      <td>1.773339e-02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Leandro</td>
      <td>chegou atrasado segunda aula</td>
      <td>22.200001</td>
      <td>18</td>
      <td>2.0</td>
      <td>1.049763e-06</td>
      <td>0.011822</td>
      <td>9.881774e-01</td>
      <td>2</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Estela</td>
      <td>nao chegou atrasado</td>
      <td>1.000000</td>
      <td>13</td>
      <td>0.0</td>
      <td>9.999809e-01</td>
      <td>0.000019</td>
      <td>5.741634e-11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 9 columns</p>
</div>



Agora cada observação conta com a presença das colunas 0,1 e 2, que mostram a probabilidade para cada evento.
E também com a coluna 'predição' que mostra o evento com maior probabilidade de ocorrência.

### Criando Variavel "Predição" que rotula o evento com maior probabilidade


```python
dados_multinomial.loc[dados_multinomial['predicao']==0,
                            'predicao_label'] ='nao chegou atrasado'
dados_multinomial.loc[dados_multinomial['predicao']==1,
                            'predicao_label'] ='chegou atrasado primeira aula'
dados_multinomial.loc[dados_multinomial['predicao']==2,
                            'predicao_label'] ='chegou atrasado segunda aula'


```

### Visualizando Predições Rotuladas


```python
dados_multinomial
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estudante</th>
      <th>atrasado</th>
      <th>dist</th>
      <th>sem</th>
      <th>atrasado2</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>predicao</th>
      <th>predicao_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gabriela</td>
      <td>chegou atrasado segunda aula</td>
      <td>20.500000</td>
      <td>15</td>
      <td>2.0</td>
      <td>1.801024e-02</td>
      <td>0.523388</td>
      <td>4.586018e-01</td>
      <td>1</td>
      <td>chegou atrasado primeira aula</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Patricia</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.299999</td>
      <td>18</td>
      <td>2.0</td>
      <td>2.751301e-06</td>
      <td>0.018737</td>
      <td>9.812605e-01</td>
      <td>2</td>
      <td>chegou atrasado segunda aula</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gustavo</td>
      <td>chegou atrasado segunda aula</td>
      <td>21.400000</td>
      <td>16</td>
      <td>2.0</td>
      <td>6.796190e-04</td>
      <td>0.173472</td>
      <td>8.258489e-01</td>
      <td>2</td>
      <td>chegou atrasado segunda aula</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Leticia</td>
      <td>chegou atrasado segunda aula</td>
      <td>31.500000</td>
      <td>15</td>
      <td>2.0</td>
      <td>2.759476e-07</td>
      <td>0.003748</td>
      <td>9.962518e-01</td>
      <td>2</td>
      <td>chegou atrasado segunda aula</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Luiz Ovidio</td>
      <td>chegou atrasado segunda aula</td>
      <td>17.500000</td>
      <td>16</td>
      <td>2.0</td>
      <td>2.083782e-02</td>
      <td>0.601588</td>
      <td>3.775739e-01</td>
      <td>1</td>
      <td>chegou atrasado primeira aula</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Sheila</td>
      <td>nao chegou atrasado</td>
      <td>24.000000</td>
      <td>10</td>
      <td>0.0</td>
      <td>9.531361e-01</td>
      <td>0.046317</td>
      <td>5.471598e-04</td>
      <td>0</td>
      <td>nao chegou atrasado</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Pedro</td>
      <td>chegou atrasado primeira aula</td>
      <td>15.000000</td>
      <td>15</td>
      <td>1.0</td>
      <td>4.146183e-01</td>
      <td>0.557343</td>
      <td>2.803830e-02</td>
      <td>1</td>
      <td>chegou atrasado primeira aula</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Horacio</td>
      <td>chegou atrasado primeira aula</td>
      <td>14.400000</td>
      <td>15</td>
      <td>1.0</td>
      <td>5.008256e-01</td>
      <td>0.481441</td>
      <td>1.773339e-02</td>
      <td>0</td>
      <td>nao chegou atrasado</td>
    </tr>
    <tr>
      <th>98</th>
      <td>Leandro</td>
      <td>chegou atrasado segunda aula</td>
      <td>22.200001</td>
      <td>18</td>
      <td>2.0</td>
      <td>1.049763e-06</td>
      <td>0.011822</td>
      <td>9.881774e-01</td>
      <td>2</td>
      <td>chegou atrasado segunda aula</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Estela</td>
      <td>nao chegou atrasado</td>
      <td>1.000000</td>
      <td>13</td>
      <td>0.0</td>
      <td>9.999809e-01</td>
      <td>0.000019</td>
      <td>5.741634e-11</td>
      <td>0</td>
      <td>nao chegou atrasado</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 10 columns</p>
</div>



## Métricas de Análise Do Modelo

### Matriz de Confusão


```python
labels = ['não chegou atrasado',
          'chegou atrasado na primeira aula',
          'chegou atrasado na segunda aula']

matrix = confusion_matrix(dados_multinomial['atrasado'],
                          dados_multinomial['predicao_label'])

pd.DataFrame(matrix, 
             index= labels, 
             columns = labels)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>não chegou atrasado</th>
      <th>chegou atrasado na primeira aula</th>
      <th>chegou atrasado na segunda aula</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>não chegou atrasado</th>
      <td>12</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>chegou atrasado na primeira aula</th>
      <td>5</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>chegou atrasado na segunda aula</th>
      <td>2</td>
      <td>0</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
</div>


A matriz de confusão mostra a taxa de acerto para cada evento. Não existe a presença de especifidade, sensitivdade para regressões multinomiais.
Não chegou atrasado - taxa de acerto 12/16 eventos.
chegou atrasado na primeira aula - taxa de acerto 30/35 eventos.
chegou atrasado na segunda aula - taxa de acerto 47/49 eventos.
### Acurácia Global


```python
table = pd.pivot_table(dados_multinomial,
                       index=['predicao_label'],
                       columns=['atrasado'],
                       aggfunc='size')
table = table.fillna(0)
table = table.to_numpy()
acuracia = table.diagonal().sum()/table.sum()
acuracia
```




    0.89


O modelo apresentou uma acurácia global de 89%, 89/100 acertos.