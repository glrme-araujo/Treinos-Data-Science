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


## Estatisticas Descritivas

summary(dataset)



## A coluna Species esta como character, vou trocar para factor.

dataset$Species <- as.factor(dataset$Species)



## Visualizar as primeiras linhas do dataset, facilita entendimento.

head(dataset) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)


# Plotar quantidade de amostras por especie
ggplot(data= dataset) +
      geom_bar(mapping = aes(x = Species,fill=Species))+
      labs( y= "Quantidade", x = "Especies")



# Filtar por especie

Iris_setosa     <- filter(dataset,Species =='Iris-setosa')

Iris_versicolor <- filter(dataset,Species =='Iris-versicolor')      

Iris_virginica  <- filter(dataset,Species =='Iris-virginica') 



## Gerando data frame com valores medios de cada especie

valores_medios <- data.frame(
  
  row.names = c("Iris-setosa","Iris-versicolor","Iris-virginica"),
  
  SepalLengthCm = c(mean(Iris_setosa$SepalLengthCm),
                    mean(Iris_versicolor$SepalLengthCm),
                    mean(Iris_virginica$SepalLengthCm)),
  
  SepalWidthCm = c(mean(Iris_setosa$SepalWidthCm),
                   mean(Iris_versicolor$SepalWidthCm),
                   mean(Iris_virginica$SepalWidthCm)),
  
  PetalLengthCm = c(mean(Iris_setosa$PetalLengthCm),
                    mean(Iris_versicolor$PetalLengthCm),
                    mean(Iris_virginica$PetalLengthCm)),
  
  PetalWidthCm = c(mean(Iris_setosa$PetalWidthCm),
                   mean(Iris_versicolor$PetalWidthCm),
                   mean(Iris_virginica$PetalWidthCm))
  
  
)







valores_medios %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)

############### Gerando Radar para cada  especie ##################



### Gerando Data frame com valores maximos e minimos para cada variavel

max_min <- data.frame(
  
  row.names = c("Max", "Min"),
  
  SepalLengthCm = c(max(dataset$SepalLengthCm),
                    min(dataset$SepalLengthCm)),
  
  SepalWidthCm = c(max(dataset$SepalWidthCm),
                   min(dataset$SepalWidthCm)),
  
  PetalLengthCm = c(max(dataset$PetalLengthCm),
                    min(dataset$PetalLengthCm)),
  
  PetalWidthCm = c(max(dataset$PetalWidthCm),
                   min(dataset$PetalWidthCm))
)








#### FUnção para gerar Radar elaborado

create_beautiful_radarchart <- function(data, color = "#00AFBB", 
                                        vlabels = colnames(data), vlcex = 0.7,
                                        caxislabels = NULL, title = NULL, ...){
  radarchart(
    data, axistype = 1,
    # Customize the polygon
    pcol = color, pfcol = scales::alpha(color, 0.5), plwd = 2, plty = 1,
    # Customize the grid
    cglcol = "grey", cglty = 1, cglwd = 0.8,
    # Customize the axis
    axislabcol = "grey", 
    # Variable labels
    vlcex = vlcex, vlabels = vlabels,
    caxislabels = caxislabels, title = title, ...
  )
}

# Gerando Data frama com valores maxímos, mínimos, médios e Títulos

df <- rbind(max_min, valores_medios)
df 

# Definindo Cores e Titulos

colors <- c("#00AFBB", "purple", "#FC4E07")
titles <- c("Iris-setosa","Iris-versicolor","Iris-virginica")



# Reduzindo e dividindo espaço para 3 plotagens

op <- par(mar = c(1, 1, 1, 1))

par(mfrow = c(1,3))

# Plotando Radares Individuais

for(i in 1:3){
  create_beautiful_radarchart(
    data = df[c(1, 2, i+2), ], caxislabels = c(0, 2, 5, 8, 10),
    color = colors[i], title = titles[i]
  )
  
}


# Reduzindo Area de plotagem
op <- par(mar = c(1, 2, 2, 2))



# Plotando Radar Sobreposto
create_beautiful_radarchart(
  data = df, caxislabels = c(0, 5, 10, 15, 20),
  color = c("#00AFBB", "purple", "#FC4E07")
)


# Adicionando Legendas
legend(
  x = "bottom", legend = rownames(df[-c(1,2),]), horiz = TRUE,
  bty = "n", pch = 20 , col = c("#00AFBB", "purple", "#FC4E07"),
  text.col = "black", cex = 1, pt.cex = 1.5
)

############################################################################
################## Parte 3 - Logistica Multinominal #########################


# Criando Referencia para regressão


dataset$Species <- relevel(dataset$Species, 
                           ref = "Iris-virginica")

#Estimação do modelo - função multinom do pacote nnet

modelo_especies <- multinom(formula = Species ~ SepalLengthCm + 
                              SepalWidthCm +PetalLengthCm +PetalWidthCm  , 
                            data = dataset)
stargazer(modelo_especies, nobs=T, type="text")


#LL do modelo_especies
logLik(modelo_especies)


#Fazendo predições para o modelo_especie
predict(modelo_especies, 
        data.frame(SepalLengthCm= 2,
                   SepalWidthCm=2,
                   PetalLengthCm =2,
                   PetalWidthCm=2), 
        type = "probs")


predict(modelo_especies, 
        data.frame(SepalLengthCm= 2,
                   SepalWidthCm=2,
                   PetalLengthCm =2,
                   PetalWidthCm=2), 
        type = "class")



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
EGM <- as.data.frame.matrix(table(predicao, Species))

#Visualizando a tabela EGM
EGM %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = F, 
                font_size = 22)

#Eficiência global do modelo
acuracia <- (round((sum(diag(table(Species, predicao))) / 
                      sum(table(Species, predicao))), 2))

acuracia

# vemos que o modelo sem analise fatorial tem um acuracia de 0.99 ou 99%

###############################################################################
################## Parte 4 - Análise Fatorial PCA #########################

# Estabelecendo uma matriz de correlações de Pearson

rho <- cor(dataset[,2:5])


## Plotando Grafico de Calor das Correções
rho %>% 
  melt() %>% 
  rename(Correlação = value) %>%
  ggplot() +
  geom_tile(aes(x = Var1, y = Var2, fill = Correlação)) +
  geom_text(aes(x = Var1, y = Var2, label = format(Correlação, digits = 2)),
            size = 3) +
  scale_fill_gradient2(low = "dodgerblue4", 
                       mid = "white", 
                       high = "brown",
                       midpoint = 0) +
  labs(x = NULL, y = NULL) +
  theme(panel.background = element_rect("white"),
        panel.grid = element_line("grey95"),
        panel.border = element_rect(NA),
        legend.position = "bottom")


# Teste da adequabilidade dos dados para PCA (Teste de esfericidade de Bartlett)
cortest.bartlett(dataset[,2:5])

# Quiquadrado alto e P-valor menor que 0.05. Existe Significancia estatistica



# Análise Fatorial por Componentes Principais (PCA)

fatorial <- principal(dataset[,2:5],
                      nfactors = length(dataset[,2:5]),
                      rotate = "none",
                      scores = TRUE)

fatorial



# Eigenvalues (autovalores)
eigenvalues <- round(fatorial$values, 5)
eigenvalues




# Identificação da variância compartilhada em cada fator
variancia_compartilhada <- as.data.frame(fatorial$Vaccounted) %>% 
  slice(1:3)

rownames(variancia_compartilhada) <- c("Autovalores",
                                       "Prop. da Variância",
                                       "Prop. da Variância Acumulada")




# Variância compartilhada pelas variáveis originais para a formação de cada fator
round(variancia_compartilhada, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)




# Cálculo dos scores fatoriais
scores_fatoriais <- as.data.frame(fatorial$weights)





# Visualização dos scores fatoriais
round(scores_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)





# Cálculo dos fatores propriamente ditos
fatores <- as.data.frame(fatorial$scores)

View(fatores)



# Cálculo das cargas fatoriais
cargas_fatoriais <- as.data.frame(unclass(fatorial$loadings))




# Visualização das cargas fatoriais - Correlações entre fator e variavel orig.
round(cargas_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)





# Cálculo das comunalidades
comunalidades <- as.data.frame(unclass(fatorial$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades (aqui são iguais a 1 para todas as variáveis)
# Foram extraídos 4 fatores neste primeiro momento
round(comunalidades, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)

### Elaboração da Análise Fatorial por Componentes Principais ###

### Fatores extraídos a partir de autovalores maiores que 1 ###

# Definição da quantidade de fatores com eigenvalues maiores que 1
k <- sum(eigenvalues > 1)
print(k)


# Elaboração da análise fatorial por componentes principais
# Com quantidade 'k' de fatores com eigenvalues maiores que 1
# Utilizarei 2, pois o segundo fator tem um valor proximo de 1 ( 0,92)

fatorial2 <- principal(dataset[,2:5],
                       nfactors = 2,
                       rotate = "none",
                       scores = TRUE)
fatorial2




# Cálculo das comunalidades com apenas os 'k' ('k' = 2) primeiros fatores
comunalidades2 <- as.data.frame(unclass(fatorial2$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades com apenas os 'k' ('k' = 2) primeiros fatores
round(comunalidades2, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)

# Loading plot com as cargas dos 'k' ('k' = 2) primeiros fatores
cargas_fatoriais[, 1:2] %>% 
  data.frame() %>%
  rownames_to_column("variáveis") %>%
  ggplot(aes(x = PC1, y = PC2, label = variáveis)) +
  geom_point(color = "darkorchid",
             size = 3) +
  geom_text_repel() +
  geom_vline(aes(xintercept = 0), linetype = "dashed", color = "orange") +
  geom_hline(aes(yintercept = 0), linetype = "dashed", color = "orange") +
  expand_limits(x= c(-1.25, 0.25), y=c(-0.25, 1)) +
  theme_bw()




# Adicionando os fatores extraídos no banco de dados original
dataset <- bind_cols(dataset,
                     "fator1" = fatores$PC1, 
                     "fator2" = fatores$PC2)




# Criação de um ranking Critério da soma ponderada e ordenamento)
# O ranking auxilia na busca e analise de cada amostra


dataset$ranking <- fatores$PC1 * variancia_compartilhada$PC1[2] +
  fatores$PC2 * variancia_compartilhada$PC2[2]

# Visualizando o ranking final
dataset %>%
  arrange(desc(ranking)) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 12)

# Final da Analise Fatorial #

################################################################################ 
####################### PARTE FINAL - COMPARAÇÃO ENTRE MODELOS ################

# Primeiro vamos estimar o modelo com analise fatorial
# Criando Referencia para regressão
# vamos fazer um novo dataset para podermos comparar

dataset_comparacao <- dataset


dataset_comparacao$Species <- relevel(dataset_comparacao$Species, 
                           ref = "Iris-virginica")

#Estimação do modelo - função multinom do pacote nnet

modelo_especies_comp <- multinom(formula = Species ~ fator1 + fator2 , 
                            data = dataset_comparacao)
stargazer(modelo_especies_comp, nobs=T, type="text")


#LL do modelo_especies
logLik(modelo_especies_comp)




#A EFETIVIDADE GERAL DO MODELO               

#Adicionando as prováveis ocorrências de evento apontadas pela modelagem à 
#base de dados

dataset_comparacao$predicao <- predict(modelo_especies_comp, 
                            newdata = dataset_comparacao, 
                            type = "class")

#Visualizando a nova base de dados dataset com a variável 'predicao'
dataset_comparacao %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = F, 
                font_size =12)

attach(dataset_comparacao)

#Criando uma tabela para comparar as ocorrências reais com as predições
EGM_comp <- as.data.frame.matrix(table(predicao, Species))

#Visualizando a tabela EGM
EGM_comp %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = F, 
                font_size = 22)

#Eficiência global do modelo
acuracia <- (round((sum(diag(table(Species, predicao))) / 
                      sum(table(Species, predicao))), 2))

acuracia


## O modelo estimado com analise fatorial tem uma acuracia de 0.92 ou 92%
