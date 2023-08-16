# Carregar mapa do brasil
mapa <- read_country(year =2020)

# Trocar Nomes para Factor
mapa$name_region <- as.factor(mapa$name_region)

# Trocando as regiões
levels(mapa$name_region) <- list(N = "Norte",
                                 NE = "Nordeste",
                                 Sul = "S",
                                 CO = "Centro Oeste",
                                 SE = "Sudeste")
# Filtrando dados de 2022
dados_2022 <- filter(dataset, Data >= "2022-01-01" & Data < "2023-01-01")
             

dados_2022 <- dados_2022 %>%
              group_by(regiao)

teste <- dados_2022 %>%
      summarise(
        media = mean(Consumo)
        
      )


dados_2020 <- NULL



dados_br <- mapa %>%
  left_join(medias_2022, 
            by = c("name_region"="regiao") )
  
dados_br <- NULL
dados_br$media <- round(dados_br$media,0)

glimpse(mapa)

options(scipen = 999)

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
      labs(title="Receita anual 2022", 
           size=8) +
      xlab(" ")+
      ylab(" ")+
      theme_minimal()
