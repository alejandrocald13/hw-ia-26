# Se requiere la instalacion de los siguientes paquetes
# install.packages(c('tidyverse','caret','neuralnet','palmerpenguins'))

# Cargar paquetes
library(tidyverse)
library(caret)
library(neuralnet)
library(palmerpenguins)

# Cargar el conjunto de datos
datos = penguins %>%
  select(species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) %>%
  drop_na()

# Separacion de los datos en conjunto de entrenamiento y pruebas
muestra = createDataPartition(datos$species, p=0.8, list = FALSE)
train = datos[muestra,]
test = datos[-muestra,]

# Analisis exploratorio
head(train,5)
tail(train,5)
train[17:25,]

bill_length = train$bill_length_mm
hist(bill_length)

# Variables de entrada
xcols = c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")

# Convertir species a variables dummy
dummies = model.matrix(~ species - 1, data = train)
colnames(dummies) = gsub("species", "", colnames(dummies))

# Normalizacion min-max con base en train
mins = apply(train[, xcols], 2, min)
maxs = apply(train[, xcols], 2, max)

train.scaled = as.data.frame(scale(train[, xcols], center = mins, scale = maxs - mins))
test.scaled  = as.data.frame(scale(test[, xcols], center = mins, scale = maxs - mins))

# Dataset para entrenar red neuronal
train.nn = cbind(train.scaled, dummies)

# Entrenamiento de la red neuronal
red.neuronal = neuralnet(
  Adelie + Chinstrap + Gentoo ~ bill_length_mm + bill_depth_mm + flipper_length_mm + body_mass_g,
  data = train.nn,
  hidden = c(2,3),
  linear.output = FALSE
)

red.neuronal$act.fct

# Visualizacion de la red
plot(red.neuronal)

# Aplicar la red neuronal al conjunto de pruebas
prediccion = compute(red.neuronal, test.scaled)

# Decodificar maximo = Especie
specie.decod = apply(prediccion$net.result, 1, which.max)

specie.pred = data.frame(specie.decod)
specie.pred = mutate(
  specie.pred,
  especie = recode(specie.pred$specie.decod,
                   "1"="Adelie",
                   "2"="Chinstrap",
                   "3"="Gentoo")
)

test$species.pred = specie.pred$especie

# Ver resultados
head(test)
tail(test)

# Matriz de confusion
confusionMatrix(
  as.factor(test$species.pred),
  as.factor(test$species)
)

# Guardar conjunto de prueba para usarlo en Python
write.csv(test, "test_penguins.csv", row.names = FALSE)

# Mostrar pesos de la red
red.neuronal$weights

# Mostrar mins y maxs para usarlos en Python
mins
maxs

