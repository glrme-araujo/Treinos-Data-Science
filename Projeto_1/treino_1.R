#### TREINO DE DSA 1 ####


# DATASET IRIS - conjunto de dados sobre flores, com dimensões e especies 



################# INSTALAÇÃO DOS PACOTES #######################


pacotes <- c("ggplot2",
              "plotly", #plataforma gráfica
             "tidyverse", #carregar outros pacotes do R
             "ggrepel", #geoms de texto e rótulo para 'ggplot2' que ajudam a
             #evitar sobreposição de textos
             "knitr", "kableExtra", #formatação de tabelas
             "reshape2", #função 'melt'
             "misc3d", #gráficos 3D
             "plot3D", #gráficos 3D
             "cluster", #função 'agnes' para elaboração de clusters hierárquicos
             "factoextra", #função 'fviz_dend' para construção de dendrogramas
             "psych", #
             "PerformanceAnalytics",
             'Hmisc',
             'nnet',
             'stargazer',
             "ade4") #função 'ade4' para matriz de distâncias em var. binárias

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

######################################################


## Carregando dataset
dataset <- read.csv("Iris.csv",
                    dec ='.',
                    sep = ',')


### Visualização do dataset
view(dataset)


## Estatisticas descritivas
summary(dataset)


## A coluna Species esta como character, vou trocar para factor.
dataset$Species <- as.factor(dataset$Species)

## Inicio -  Analise Fatorial


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

  
  
# Análise Fatorial por Componentes Principais (PCA)
  fatorial <- principal(dataset[,2:5],
                        nfactors = length(dataset[,2:5]),
                        rotate = "none",
                        scores = TRUE)

  fatorial
# Eigenvalues (autovalores)
  eigenvalues <- round(fatorial$values, 5)
  eigenvalues
  
  
# Soma dos eigenvalues = 4 (quantidade de variáveis na análise)
# Também representa a quantidade máxima de possíveis fatores na análise
  round(sum(eigenvalues), 2)
  
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
  
  # Coeficientes de correlação de Pearson para cada par de fatores (ortogonais)
  rho <- rcorr(as.matrix(fatores), type="pearson")
  round(rho$r, 4)
  
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
  dataset$ranking <- fatores$PC1 * variancia_compartilhada$PC1[2] +
    fatores$PC2 * variancia_compartilhada$PC2[2]
  
  # Visualizando o ranking final
  dataset %>%
    arrange(desc(ranking)) %>%
    kable() %>%
    kable_styling(bootstrap_options = "striped",
                  full_width = FALSE,
                  font_size = 17)
  
  # Final da Analise Fatorial #
  
################################################################################  
################################################################################ 
  
  #####################  Regressão Multinominal  #######################
  
  # Criando Referencia para regressão
  
  
  dataset$Species <- relevel(dataset$Species, 
                            ref = "Iris-virginica")
    
  
  
  #Estimação do modelo - função multinom do pacote nnet
  modelo_especies <- multinom(formula = Species ~ SepalLengthCm + 
                              SepalWidthCm +PetalLengthCm +PetalWidthCm  , 
                              data = dataset)
  stargazer(modelo_especies, nobs=T, type="text")
  
  #LL do modelo_atrasado
  logLik(modelo_especies)
  
  #A função summ do pacote jtools não funciona para objetos de classe 'multinom'. Logo,
  #vamos definir uma função Qui2 para se extrair a estatística geral do modelo:
  Qui2 <- function(x) {
    maximo <- logLik(x)
    minimo <- logLik(update(x, ~1, trace = F))
    Qui.Quadrado <- -2*(minimo - maximo)
    pvalue <- pchisq(Qui.Quadrado, df = 1, lower.tail = F)
    df <- data.frame()
    df <- cbind.data.frame(Qui.Quadrado, pvalue)
    return(df)
  }
  
  #Estatística geral do modelo_atrasado
  Qui2( modelo_especies)
  
  #Na Regressão Logística Multinomial, o R quebra a lógica de relatórios que, 
  #normalmente, oferece para os GLM. Também é preciso notar que a linguagem 
  #básica  não consegue rodar esse tipo de regressão, sendo necessário o pacote
  #nnet. Além do mais, não são fornecidas as estatísticas z de Wald, nem os
  #p-values das variáveis da modelagem.
  
  #Explicando a lógica do R para a Logística Multinomial:
  
  #1 - Foram estabelecidas *labels* para as categorias da variável dependente: 
  #'nao chegou atrasado', 'chegou atrasado à primeira aula' e 'chegou atrasado à
  #segunda aula';
  
  #2 - Foi comandado que a categoria de referência seria a categoria 'nao chegou
  #atrasado', e isso explica o porquê dela não aparecer no relatório gerado;
  
  #3 - O relatório é dividido em duas partes: 'Coefficients' e 'Std. Errors'. 
  #Cada linha da seção 'Coefficients' informa um logito para cada categoria da
  #variável dependente, com exceção da categoria de referência. Já a seção 
  #'Std. Errors' informa o erro-padrão de cada parâmetro em cada logito.
  
  #Para calcular as estatísticas z de Wald, há que se dividir os valores da 
  #seção 'Coefficients' pelos valores da seção 'Std. Errors.' Assim, temos que:  
  
  zWald_modelo_especies <- (summary( modelo_especies)$coefficients / 
                              summary( modelo_especies)$standard.errors)
  
  zWald_modelo_especies
  
  #Porém, ainda faltam os respectivos p-values. Assim, os valores das probabilidades 
  #associadas às abscissas de uma distribuição normal-padrão é dada pela função
  #pnorm(), considerando os valores em módulo - abs(). Após isso, multiplicamos 
  #por dois os valores obtidos para considerar os dois lados da distribuição
  #normal padronizada (distribuição bicaudal). Desta forma, temos que:
  round((pnorm(abs(zWald_modelo_especies), lower.tail = F) * 2), 4)

  #Fazendo predições para o modelo_atrasado. Exemplo: qual a probabilidade média
  #de atraso para cada categoria da variável dependente, se o indivíduo tiver 
  #que percorrer 22km e passar por 12 semáforos?
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
  ##############################################################################
  #                   EXEMPLO 04 - A EFETIVIDADE GERAL DO MODELO               #
  ##############################################################################
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
                  font_size = 22)
  
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
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  