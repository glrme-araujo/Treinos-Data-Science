################################################################

# Instalação e Carregamento de Todos os Pacotes ---------------------------

pacotes <- c("readxl","plotly","tidyverse","gridExtra","forecast","TTR",
             "smooth","tidyverse", "tsibble", "fable","tsibbledata", "fpp3",
             "urca","geobr")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T)
} else {
  sapply(pacotes, require, character = T)
}






# Série de Consumo de Energia Elétrica GWh
# Regiões CO, N, NE, S, SE (529 para cada região)
# 2645 observações
# janeiro de 1979 até janeiro de 2023


dataset <- read_excel("consumo_brasil.xlsx")

####### Analise Exploratoria

summary(dataset)

# Transformando região em factor

dataset$regiao <- as.factor(dataset$regiao)


# Transformando data em data(reconhecivel pelo R)

dataset$Data <- as.Date(paste0(gsub("\\.", "-", dataset$Data),"-","01"))





# Variaveis descritivas
summary(dataset)

# Graficos por Região

##### Região Sudeste

SE <-filter(dataset,regiao == "SE")
ggplotly(
ggplot(SE,
       mapping = aes(Data,Consumo,color = regiao))+
       geom_line())

###### Região Centro-Oeste

CO <-filter(dataset,regiao == "CO")
ggplotly(
  ggplot(CO,
         mapping = aes(Data,Consumo,color = regiao))+
         geom_line())

#### Região Norte

N <-filter(dataset,regiao == "N")
ggplotly(
  ggplot(N,
         mapping = aes(Data,Consumo,color = regiao))+
         geom_line())

####### Região Nordeste
NE <-filter(dataset,regiao == "NE")
ggplotly(
  ggplot(NE,
         mapping = aes(Data,Consumo,color = regiao))+
         geom_line())


### Região Sul
S <-filter(dataset,regiao == "S")
ggplotly(
  ggplot(N,
         mapping = aes(Data,Consumo,color = regiao))+
         geom_line())


# Graficos de todas as regiões

ggplotly(
  dataset %>%
    ggplot(mapping = aes(Data,Consumo,color= regiao, group = regiao))+
    geom_line())



##################### Plotando dados de 2022 no mapa ########################

# Carregar mapa do brasil
mapa <- read_country(year =2020)

# Trocar Nomes para Factor
mapa$name_region <- as.factor(mapa$name_region)

# Trocando os nomes das regiões
levels(mapa$name_region) <- list(N = "Norte",
                                 NE = "Nordeste",
                                 S = "Sul",
                                 CO = "Centro Oeste",
                                 SE = "Sudeste")
# Filtrando dados de 2022
dados_2022 <- filter(dataset, Data >= "2022-01-01" & Data < "2023-01-01")

# Agrupando por Região
dados_2022 <- dados_2022 %>%
  group_by(regiao)



# Tirando as medias dos grupos
medias_2022 <- dados_2022 %>%
  summarise(
    media = mean(Consumo))


# Realizando um merge das medias com os dados do mapa
dados_br <- mapa %>%
  left_join(medias_2022, 
            by = c("name_region"="regiao") )

# Arredondando N de casas decimais da media
dados_br$media <- round(dados_br$media,0)


# Plotando Mapa
ggplot()+
  geom_sf(data=dados_br,
          aes(fill=media),
          color = "grey")+
  geom_sf_text(data=dados_br,
               aes(label=media),
               size=3,
               color ="white")+
  scale_fill_gradient(low = "blue", 
                      high = "red",
                      name="Total (GWh)") +
  labs(title="Consuno anual - 2022", 
       size=8) +
  xlab(" ")+
  ylab(" ")+
  theme_minimal()

###########################################################