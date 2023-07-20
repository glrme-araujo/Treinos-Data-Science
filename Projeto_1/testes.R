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
  