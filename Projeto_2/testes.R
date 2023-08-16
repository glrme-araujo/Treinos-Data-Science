dataset$Data <- as.Date(paste0(gsub("\\.", "-", dataset$Data),"-","01"))
as.Date(paste0(gsub("\\.", "-", Data),"-","01"))
data3 <- as.Date(gsub("\\.","-",dataset$Data))


datateste <- dataset$Data
datateste <- gsub("\\.","-",datateste)
datateste <- paste0(datateste,"-","01")
datateste<-as.Date(datateste)
teste <- ts(dataset$Consumo)
teste2 <- dataset  %>% group_by(regiao)


plot(ts(teste2$Data),
     col = teste2$regiao
     )




ggplot(teste2)+
  geom_line(mapping = aes(x = Data, 
                          y = Consumo,
                          color = regiao))+
  facet_wrap(~regiao, nrow=5)+
  scale_y_continuous()
  glimpse(dataset)


reg <-filter(dataset,regiao == "SE")

teste <- filter
print(length(levels(dataset$regiao)))
par(mfrow = c(2,2))

for( i in levels(dataset$regiao)){
  selecao <- filter(dataset,regiao == i)
  

}


dada <- t(dataset)
