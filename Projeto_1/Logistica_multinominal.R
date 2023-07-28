# Projeto 1 - Script "Logistica Multinominal"


# Dataset - Iris.csv
# A base de dados conta com 150 amostras de 3 especies de flores
# Além de especies temos dimensões das petalas e setalas.



# O script tem como objetivo realizar a predição das espécies 
# Utilizando modelos de logística.


###########################################################################

################## Parte 1 - Instalação dos Pacotes ######################



pacotes <- c("plotly","tidyverse","knitr","kableExtra","fastDummies","rgl","car",
             "reshape2","jtools","stargazer","lmtest","caret","pROC","ROCR","nnet",
             "magick","cowplot","globals","equatiomatic")

options(rgl.debug = TRUE)

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

################## Parte 2 - Análise Introdutória #########################




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



# Adicionar a coluna da unidade

unidade <- 'cm'

dataset <- mutate(dataset,unidade)%>%
  relocate(unidade, .before =Species)



## Alterar os Nomes das colunas

novos_nomes <- c("Observação",
                 "Compr.Sepala",
                 "Larg.Sepala",
                 "Compr.Petala",
                 "Larg.Petala",
                 "Unidade",
                 "Espécie")

names(dataset) <- novos_nomes

## Visualizar as duas primeiras linhas do dataset, facilita entendimento.

head(dataset,n =2) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)


# Criando Referencia para regressão

dataset$Espécie <- relevel(dataset$Espécie, 
                           ref = "Iris-virginica")



#Estimação do modelo - função multinom do pacote nnet

modelo_especies <- multinom(formula = Espécie ~ 
                              Compr.Sepala + 
                              Larg.Sepala	 +
                              Compr.Petala +
                              Larg.Petala, 
                            data = dataset)
#LL do modelo_especies
# quanto maior, melhor

logLik(modelo_especies)



#A EFETIVIDADE GERAL DO MODELO               

#Adicionando as prováveis ocorrências de evento apontadas pela modelagem à 
#base de dados

dataset$predicao <- predict(modelo_especies, 
                            newdata = dataset, 
                            type = "class")

#Visualizando a nova base de dados dataset com a variável 'predicao'
dataset %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = F, 
                font_size =12)

attach(dataset)

#Criando uma tabela para comparar as ocorrências reais com as predições
confusao <- as.data.frame.matrix(table(predicao, Espécie))

#Visualizando a tabela EGM
confusao %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = F, 
                font_size = 22)


#Eficiência global do modelo
acuracia <- (round((sum(diag(table(Espécie, predicao))) / 
                      sum(table(Espécie, predicao))), 2))

acuracia





