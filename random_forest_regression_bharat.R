#Random Forest Regression

#Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]
# 
# #Splitting the dataset into training set and test set
# #install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary,
#                      SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# #Feature Scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])


#Fitting Regression MOdel to the dataset
#Create your regressor here
install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)


#Predicting a new result with Polynomial regression
y_pred = predict(regressor, data.frame(Level = 6.5))




#Visualising the plot on regression Model(for higher resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y= dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, 
                                        newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle("Random Forest Model") +
  xlab("Level") +
  ylab("Salary")
