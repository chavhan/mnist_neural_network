############# Image classification on MNIST data set ##########
library(keras)
mnist <- dataset_mnist()
str(mnist)
# List of 2
# $ train:List of 2
# ..$ x: int [1:60000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
# ..$ y: int [1:60000(1d)] 5 0 4 1 9 2 1 3 1 4 ...
# $ test :List of 2
# ..$ x: int [1:10000, 1:28, 1:28] 0 0 0 0 0 0 0 0 0 0 ...
# ..$ y: int [1:10000(1d)] 7 2 1 0 4 1 4 9 5 9 ...

trainx <- mnist$train$x
trainy <- mnist$train$y
testx <- mnist$test$x
testy <- mnist$test$y

table(mnist$train$y,mnist$train$y)


### it is showing how many images we have in train y matrix 

#      0    1    2    3    4    5    6    7    8    9
# 0 5923    0    0    0    0    0    0    0    0    0
# 1    0 6742    0    0    0    0    0    0    0    0
# 2    0    0 5958    0    0    0    0    0    0    0
# 3    0    0    0 6131    0    0    0    0    0    0
# 4    0    0    0    0 5842    0    0    0    0    0
# 5    0    0    0    0    0 5421    0    0    0    0
# 6    0    0    0    0    0    0 5918    0    0    0
# 7    0    0    0    0    0    0    0 6265    0    0
# 8    0    0    0    0    0    0    0    0 5851    0
# 9    0    0    0    0    0    0    0    0    0 5949

## showing the bitmap image of x matrix for starting 9 elements 
par(mfrow = c(3,3))
for(i in 1:9)
  plot(as.raster(trainx[i,,], max = 255))

hist(trainx[1,,])  ### it helps us to understand how many number is being used to make the images five
##  which very from 0 to 254 
View(trainx[1,,])

####### If we want to look different images of five number 
five_images <- c(1,12,36,48,66)
par(mfrow = c(2,3))
for(i in five_images) plot(as.raster(trainx[i,,],max = 255))

### reshaping the trainx matrix and converting it to on dimentions rather than to 28 by 28 
trainx <- array_reshape(trainx,c(nrow(trainx), 784))
str(trainx)
##num [1:60000, 1:784] 0 0 0 0 0 0 0 0 0 0 ...
View(trainx[1,])

testx <- array_reshape(testx, c(nrow(testx), 784))

## changing the value of number from 0 to 255 to 0 to 1 
trainx <- trainx/255
testx <- testx/255
hist(trainx[1,]) ### to check the bitmap number values 

### one hot encoding 
trainy <- to_categorical(trainy,10)
testy <- to_categorical(testy,10)
head(testy)
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
# [1,]    0    0    0    0    0    0    0    1    0     0
# [2,]    0    0    1    0    0    0    0    0    0     0
# [3,]    0    1    0    0    0    0    0    0    0     0
# [4,]    1    0    0    0    0    0    0    0    0     0
# [5,]    0    0    0    0    1    0    0    0    0     0
# [6,]    0    1    0    0    0    0    0    0    0     0

library(dplyr)
library(tensorflow)
## creating the model 
model <- keras_model_sequential() 
model %>% 
          layer_dense(units = 128, activation = 'relu' , input_shape = c(784)) %>% ## it is indicating we have one hidden layer with 
  ## 128 neurons and 784 input neurons and activation function is relu
          layer_dense(units = 10, activation = 'softmax')

# Model: "sequential_1"
# _____________________________________________________________________________________________________________________________________________
# Layer (type)                                                   Output Shape                                            Param #               
# =============================================================================================================================================
#   dense_1 (Dense)                                                (None, 128)                                             100480                
# _____________________________________________________________________________________________________________________________________________
# dense (Dense)                                                  (None, 10)                                              1290                  
# =============================================================================================================================================
#   Total params: 101,770
# Trainable params: 101,770
# Non-trainable params: 0
# _____________________________________________________________________________________________________________________________________________


###### Tuning the model to over come from overfitting 
model %>% 
  layer_dense(units = 128, activation = 'relu' , input_shape = c(784)) %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax') 
summary(model)
# 
# Model: "sequential_3"
# _____________________________________________________________________________________________________________________________________________
# Layer (type)                                                   Output Shape                                            Param #               
# =============================================================================================================================================
#   dense_12 (Dense)                                               (None, 128)                                             100480                
# _____________________________________________________________________________________________________________________________________________
# dropout_5 (Dropout)                                            (None, 128)                                             0                     
# _____________________________________________________________________________________________________________________________________________
# dense_11 (Dense)                                               (None, 64)                                              8256                  
# _____________________________________________________________________________________________________________________________________________
# dropout_4 (Dropout)                                            (None, 64)                                              0                     
# _____________________________________________________________________________________________________________________________________________
# dense_10 (Dense)                                               (None, 10)                                              650                   
# =============================================================================================================================================
#   Total params: 109,386
# Trainable params: 109,386
# Non-trainable params: 0
# _____________________________________________________________________________________________________________________________________________


## compile the model 
model %>% 
          compile(loss = 'categorical_crossentropy',
                  optimizer =  optimizer_rmsprop(),
                  metrics = 'accuracy'
                  )

# model %>%
#   compile(loss = "sparse_categorical_crossentropy",
#           optimizer = "adam",
#           metrics = "accuracy")

# print_dot_callback <- callback_lambda(
#   on_epoch_end = function(epoch, logs) {
#     if (epoch %% 10 == 0) cat("\n")
#     cat(".")
#   }
# )    
## Fit the model 
history <- model %>%
                    fit(
                      trainx,
                      trainy,
                      epochs = 30,
                      batch_size = 32,
                      validation_split = 0.2
                     )

# model %>%
#   fit(
#     x = trainx,
#     y = trainy,
#     epochs = 50,
#     validation_split = 0.3,
#     verbose = 0,
#     callbacks = list(print_dot_callback)
#   )

##8s 4ms/step - loss: 0.4434 - accuracy: 0.8748 - val_loss: 0.1630 - val_accuracy: 0.9526
##4s 3ms/step - loss: 0.1327 - accuracy: 0.9607 - val_loss: 0.1296 - val_accuracy: 0.9633

##save.image(file = 'mnist_ann.Rdata')

plot(history)  ## it clearly indicates that model is facing overfilling problem 

## Evaluation and prediction 
model %>% evaluate(testx,testy)
# loss  accuracy 
# 0.1795492 0.9768000 
pred <- model %>% predict_classes(testx)
table(pred,mnist$test$y)
# pred    0    1    2    3    4    5    6    7    8    9
#     0  971    0    2    0    2    2    6    3    5    2
#     1    0 1125    2    0    1    0    3    3    1    2
#     2    2    3 1011    6    3    0    0    9    3    0
#     3    1    1    2  982    0    6    1    6    7    4
#     4    0    0    1    0  949    1    5    1    3    7
#     5    2    1    0    9    0  874    3    0    8    4
#     6    1    2    2    0    3    4  936    0    0    2
#     7    1    1    9    6    3    1    0 1000    2    5
#     8    2    2    3    5    5    3    4    1  940    3
#     9    0    0    0    2   16    1    0    5    5  980

prob <- model %>% predict_proba(testx)
cbind(prob, pred, mnist$test$y)[1:6,]  ## all 6 samples are correctly predicted 

