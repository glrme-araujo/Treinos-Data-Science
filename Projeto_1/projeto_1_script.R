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
             'fmsb',
             'tibble') 

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

# Plotar quantidade de amostras por especie
ggplot(data= dataset) +
  geom_bar(mapping = aes(x = Espécie,fill=Espécie))+
  labs( y= "Quantidade", x = "Especie")



################ Parte 3 -  Wragling de dados ################



# transformando variaveis dimensionais de colunas para linhas

dataset <- dataset %>%
  gather('Larg.Petala','Larg.Sepala', key = "Largura", value = 'compri.val')%>%
  gather('Compr.Petala','Compr.Sepala', key = "Comprimento", value = 'larg.val')


# transformando coluna "observação" em factor
# vai facilitar para Agrupar

dataset$Observação <- as.factor(dataset$Observação)

# Agrupando Dataset a partir das observações

dataset <- dataset %>% 
  group_by(Observação)%>%
  arrange(.by_group=TRUE)



# Selecionando as linhas que temos os valores dimensionais
# para petala e sepalas separadas

dataset <-mutate(dataset,
                  Parte_floral = case_when(
                  Largura == 'Larg.Petala' & Comprimento == "Compr.Petala" ~ 'Petala',
                  Largura == 'Larg.Sepala' &  Comprimento == "Compr.Sepala" ~ 'Sepala'),)



# Removendo linhas com NA's

dataset <- na.omit(dataset)



# Selecionando que vamos utilizar

dataset <- select(dataset, -Largura, -Comprimento)

# Reorganizando Posições das colunas

dataset <- dataset %>% relocate(Espécie, .after= Observação)%>%
                       relocate(Parte_floral, .after= Espécie)%>%
                       relocate(Unidade, .after= larg.val)
            
# Renomeando Colunas
dataset <- rename(dataset,
        Comprimento = compri.val,
        Largura = larg.val
       )



# transformando coluna "parte_floral" em factor
# vai facilitar para Agrupar

dataset$Parte_floral <- as.factor(dataset$Parte_floral)

# Agrupamento e Ordem

dataset <- dataset %>% 
  group_by(Observação,Espécie,Parte_floral)%>%
  arrange(.by_group=TRUE)


head(dataset) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)
## desagrupar
dataset <- dataset %>% 
ungroup()





################ Parte 4 -  Análise Exploratória ################

## Plotar grafico para ver diferenças 
# dimensionais entre as especies e as partes florais

ggplot(dataset)+
  geom_point(mapping = aes(x = Comprimento, 
                           y = Largura,
                           color = Espécie))+
  facet_wrap(~Parte_floral, nrow=2)


ggplot(dataset)+
  geom_point(mapping = aes(x = Comprimento, 
                           y = Largura,
                           color = Parte_floral))+
  facet_wrap(~Espécie, nrow=3)



## agrupando por especie parte floral

dataset <- dataset %>% 
  group_by(Espécie,Parte_floral)%>%
  arrange(.by_group=TRUE)




# Calculando valores máximos, mínimos e média 

descritiva <- dataset %>%
  summarise(
    Comprimento_med = mean(Comprimento),
    Largura_med     = mean(Largura),
    
    Comprimento_min = min(Comprimento),
    Largura_min     = min(Largura),
    
    Comprimento_max = max(Comprimento),
    Largura_max     = max(Largura),
    Unidade = unidade
  )






                
                                    


                                   
