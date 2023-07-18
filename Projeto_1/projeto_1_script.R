# Projeto 1 - Analise Fatorial, Logística Multinominal



# Dataset - Iris.csv
# A base de dados conta com 150 amostras de 3 especies de flores
# Além de especies temos dimensões das petalas e setalas.



# O Projeto se concentrará em comparar dois modelos logisticos.
# Um modelo utilizando analise fatorial e outro apenas com os dados de entrada.


###########################################################################

################## Parte 1 - Instalação dos Pacotes ######################

pacotes <- c("ggplot2",
             "plotly", #plataforma gráfica
             "tidyverse", #carregar outros pacotes do R
             "ggrepel", #geoms de texto e rótulo para 'ggplot2' que ajudam a
             #evitar sobreposição de textos
             "knitr", "kableExtra", #formatação de tabelas
             "psych", # elaboração da fatorial e estatística
             "PerformanceAnalytics",# função 'chart.Correlation' para plotagem
             'Hmisc', # matriz de correlações com p-valor
             'nnet', #
             'stargazer',
             'reshape2', 
             'fmsb') 

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

############################################################################

################## Parte 2 - Análise Exploratória #########################


# Carregando Dataset

dataset <- read.csv("Iris.csv",
                    sep =',',
                    dec = '.')

## Visualizar as primeiras linhas do dataset, facilita entendimento.

head(dataset) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)


## Estatisticas Descritivas
summary(dataset)

## A coluna Species esta como character, vou trocar para factor.

dataset$Species <- as.factor(dataset$Species)

## Alterar os Nomes das colunas

novos_nomes <- c("Observações",
                 "Compr. Sépala(cm)",
                 "Larg. Sépala(cm)",
                 "Compr. Pétala(cm)",
                 "Larg. Pétala(cm)",
                 "Espécie")

names(dataset) <- novos_nomes

## Visualizar as primeiras linhas do dataset, facilita entendimento.

head(dataset,n =2) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)

# Plotar quantidade de amostras por especie
ggplot(data= dataset) +
  geom_bar(mapping = aes(x = Espécie,fill=Espécie))+
  labs( y= "Quantidade", x = "Especie")


# Filtar por especie e removendo a coluna especie

Iris_setosa     <- filter(dataset,Espécie =='Iris-setosa')
Iris_setosa <- select(Iris_setosa, -Espécie)

Iris_versicolor <- filter(dataset,Espécie =='Iris-versicolor')  
Iris_versicolor <- select(Iris_versicolor, -Espécie)

Iris_virginica  <- filter(dataset,Espécie =='Iris-virginica')
Iris_virginica <- select(Iris_virginica, -Espécie)
