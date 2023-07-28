dataset_agrupado <- group_by(dataset,Espécie)

rm(dataset_agrupado)
names(dataset_agrupado)
descritiva <- dataset_agrupado %>%
  summarise(
    média = mean(Compr.Sepala),
    max = max(Compr.Sepala),
    min = min(Compr.Sepala),
    
  
  ) 
  
dataset2 <- dataset %>%
  gather('Larg.Petala','Larg.Sepala', key = "Largura", value = 'compri.val')%>%
  gather('Compr.Petala','Compr.Sepala', key = "Comprimento", value = 'larg.val')
  
  
rm(dataset2_de)

dataset2$Observação <- as.factor(dataset2$Observação)


dataset2 <- dataset2 %>% 
  group_by(Observação)%>%
  arrange(.by_group=TRUE)

dataset2_mutato <-mutate(dataset2,
      parte_floral = case_when(
      Largura == 'Larg.Petala' & Comprimento == "Compr.Petala" ~ 'Petala',
      Largura == 'Larg.Sepala' &  Comprimento == "Compr.Sepala" ~ 'Sepala'),
      

                           
                         
  
)
ggplot(dataset)+
  geom_point(mapping = aes(x = Comprimento, 
                           y = Largura,
                           color = Parte_floral))+
  facet_wrap(~Espécie, nrow=3)





ggplot(dataset)+
  geom_point(mapping = aes(x = Comprimento, 
                           y = Largura,
                           color = Parte_floral))+
  facet_wrap(~Espécie, nrow=3)





dataset_limpo <- na.omit(dataset2_mutato)







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

table(dataset$Grupo,dataset$Espécie)                                                    

acuracia <- (round((sum(diag(table(dataset$Grupo,dataset$Espécie))) / 
                      sum(table(dataset$Grupo,dataset$Espécie))), 2))
acuracia      






fviz_nbclust(dataset_ajustado, kmeans, method = "wss", k.max = 10)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "convex",
  shape = "Species", size = 1.5,  legend = "right", ggtheme = theme_bw(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)

coeficientes <- sort(cluster_h$height, decreasing = FALSE) 
coeficientes

dista=dist(dataset_ajustado, method="euclidean")
esquema <- as.data.frame(cbind(cluster_h$merge, coeficientes))
names(esquema) <- c("Cluster1", "Cluster2", "Coeficientes")
esquema



fviz_dend(x = cluster_h)




dista.hc=agnes(x=dista, method="complete")
dataset$cluster_H <- factor(cutree(tree = dista.hc, k = 3))
dev.off()
fviz_dend(x = dista.hc, show_labels = F)

fviz_dend(x = dista.hc,
          h = 5.5,
          color_labels_by_k = F,
          rect = T,
          rect_fill = T,
          rect_border = "black",
          lwd = 1,
          show_labels = F,
          ggtheme = theme_bw())

dataset$cluster_H <- factor(cutree(tree = dista.hc, k = 3))


acuracia <- (round((sum(diag(table(dataset$cluster_H,dataset$Espécie))) / 
                      sum(table(dataset$cluster_H,dataset$Espécie))), 2))
acuracia 


table(dataset$cluster_H,dataset$Espécie)                                                    

# Plotagem

fviz_cluster(pam.res3,data=dataset_ajustado,
             palette = c( "#00AFBB", "#E7B800", "#FC4E07"),
             geom = "point",
             shape = 15,
             ellipse.type="convex",
             star.plot=TRUE,
             ggtheme=theme_minimal(),
             main = "Cluster plot"
             
)

sil.score(dataset_ajustado, nb.clus = c(2:13), nb.run = 100, iter.max = 1000,
          method = "euclidean")


fviz_silhouette(pam.res3, palette = "jco", ggtheme = theme_classic())
pam.res3 <- pam(dataset_ajustado, 3)
pam <- fviz_cluster(pam.res3, 
             data = dataset_ajustado,
             geom = 'point',
             stand = FALSE,
             title = 'PAM — CLUSTERING',
             frame.type = 'convex')

grid.arrange(pam,
             kmeans,
            heights=unit(0.8,'npc'),
            top='Algoritmos de Segmentação, clusters = 3')





kmeans<-fviz_cluster(cluster_kmeans,data=dataset_ajustado,
             palette = c( "#00AFBB", "#E7B800", "#FC4E07"),
             geom = "point",
             shape = 15,
             ellipse.type="convex",
             star.plot=TRUE,
             ggtheme=theme_minimal(),
             main = "Cluster plot"
             
)


db <- fpc::dbscan(dataset_ajustado,eps = 0.15, MinPts = 5)
#Visualize DBSCAN Clustering
fviz_cluster(db,
              data=dataset_ajustado,
              stand = FALSE,
              show.clust.cent = FALSE,
              geom = 'point',
              title = 'DBSCAN — CLUSTERING')
 

summary(modelo_especies)
