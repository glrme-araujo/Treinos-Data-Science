# Projeto 1 - Script "Cluster"


# Dataset - Iris.csv
# A base de dados conta com 150 amostras de 3 especies de flores
# Além de especies temos dimensões das petalas e setalas.



# O script tem como objetivo realizar a clusterização do dataset,
# e comparar os métodos K-means e Hierarquico



###########################################################################

################## Parte 1 - Instalação dos Pacotes ######################


pacotes <- c("plotly", #plataforma gráfica
             "tidyverse", #carregar outros pacotes do R
             "ggrepel", #geoms de texto e rótulo para 'ggplot2' que ajudam a
             #evitar sobreposição de textos
             "ggpubr",
             "knitr", "kableExtra", #formatação de tabelas
             "cluster", #função 'agnes' para elaboração de clusters hierárquicos
             "factoextra", #função 'fviz_dend' para construção de dendrogramas
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



## Alterar os Nomes das colunas

novos_nomes <- c("Observação",
                 "Compr.Sepala",
                 "Larg.Sepala",
                 "Compr.Petala",
                 "Larg.Petala",
                   "Espécie")

names(dataset) <- novos_nomes

## Visualizar as duas primeiras linhas do dataset, facilita entendimento.

head(dataset,n =2) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)


# Padronizando valores
dataset_ajustado <- scale(dataset[,2:5])

# Deixando o dataset apenas com a coluna "especie para facilitar 
# A comparação

dataset <- select(dataset, Espécie)

# Método de Elbow para identificação do número ótimo de clusters

fviz_nbclust(dataset_ajustado, 
             kmeans,
             method = "wss", 
             k.max = 10)

# Método silhouette para identificação do número ótimo de clusters

fviz_nbclust(dataset_ajustado, 
             kmeans,
             method = "silhouette", 
             k.max = 10)



# de acordo os metodos, um numero entre 2 e 4 cluster esta bom



###################################################################


################## Parte 3 - Cluster (K-means) #########################

# Definir uma 'semente' para sempre obter mesmos valores aleatórios do kmeans

set.seed(123)

# Realizando Clusterização

cluster_kmeans <- kmeans(dataset_ajustado,
                         centers = 3)



# Criando a coluna 'grupo' para ver em qual grupo cada observação se encaixa
dataset$cluster_K <- factor(cluster_kmeans$cluster)


# tabela de confusão
kable(table(dataset$cluster_K,dataset$Espécie))%>%
kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 16)

# Plotagem

fviz_cluster(cluster_kmeans,data=dataset_ajustado,
             palette = c( "#00AFBB", "#E7B800", "#FC4E07"),
             geom = "point",
             shape = 15,
             ellipse.type="convex",
             star.plot=TRUE,
             ggtheme=theme_minimal(),
             main = "Cluster plot"
             
)

################## Parte 4 - Cluster Hierárquico #########################

# Criando Matriz de Simialiridades

matriz_d = dist(dataset_ajustado, method="euclidean")

# Realizando o cluster hierarquico

cluster_h=agnes(x=matriz_d, method="complete")

# Dendrograma com visualização dos clusters 

fviz_dend(x = cluster_h,
          k = 3,
          k_colors = c("blue", "green", "deeppink"),   
          color_labels_by_k = F,
          rect = T,
          rect_fill = T,
          rect_border = "black",
          lwd = 1,
          show_labels = F,
          ggtheme = theme_bw())

# Adicionando cluster ao dataset
dataset$cluster_H <- factor(cutree(tree = cluster_h, k = 3))



################## Parte 5- Visualizando acertividade #########################


# Plotagem Cluster K-means

k_plot <- ggplot(dataset)+
  geom_bar(mapping=aes(x=cluster_K, fill= Espécie))+
  labs( y= "Quantidade", x = "Grupo")+
  ggtitle('Cluster K-means')

# Plotagem Cluster hierarquico
h_plot <-   ggplot(dataset)+
  geom_bar(aes(x=cluster_H, fill= Espécie))+
  labs( y= "Quantidade", x = "Grupo")+
  ggtitle('Cluster hierarquico')

## Plotagem simultanea
grid.arrange(k_plot,
             h_plot,
             heights=unit(1.0,'npc'))



## Visualização Numérica
summary(dataset)
             