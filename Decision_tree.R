# Download the dataset
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(readxl)

ourdataset <- read_excel("~/Desktop/ourdataset.xlsx")

# Divide the data into train and test sets (80% vs.20%)
set.seed(999)
dt = sort(sample(nrow(ourdataset), nrow(ourdataset)*.8)) # Spilt data into 80% training sets and 20% test sets
train_dataset<-ourdataset[dt,]
test_dataset<-ourdataset[-dt,]

# Decision tree modelling
tree <- rpart(bankruptcy_status ~.,
               data = train_dataset,
               method = "class")
rpart.rules(tree, cover = TRUE)
rpart.plot(tree)

pre = predict(tree,newdata = test_dataset,type = 'class')
confusion_matrix <- table(list(Prediction = pre, True = test_dataset$bankruptcy_status))
accuracy <- sum(confusion_matrix[1], confusion_matrix[4]) / sum(confusion_matrix[1:4])
confusion_matrix
print(paste0("Accurancy:", accuracy))

# DEA + DT 
newdataset <- read_excel("~/Desktop/BBC_new_classification_data.xlsx")

# Divide the data into train and test sets (80% vs.20%)
set.seed(999)
dt = sample(1:nrow(newdataset),0.8*nrow(newdataset))
train_new<-newdataset[dt,]
test_new<-newdataset[-dt,]

# Decision tree modelling
tree_new = rpart(bankruptcy_status ~.,
                 data = train_new,
                 method = "class")
rpart.rules(tree_new, cover = TRUE)
rpart.plot(tree_new)


#Prediction & Accurancy
pre_tree_test = predict(tree_new,newdata = test_new,type = 'class')
confusion_matrix_new <- table(list(Prediction = pre_tree_test, True = test_new$bankruptcy_status))
accuracy_new <- sum(confusion_matrix_new[1], confusion_matrix_new[4]) / sum(confusion_matrix_new[1:4])
confusion_matrix_new
print(paste0("Accurancy:", accuracy_new))



