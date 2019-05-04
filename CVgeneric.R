#' @title CVgeneric
#' 
#' @description for a given classifier, CVgeneric splits training set into 
#' k folds then runs model on each fold without hold out set. the function 
#' then calculates and returns cv accuracy on hold out set along with accuracy 
#' that the kth model has on the full test set.
#' 
#' @param classifier, type of classifier -- compatible with our four methods:
#' glm (logistic regression)
#' lda (linear discriminant analysis)
#' qda (quadratic discriminant analysis)
#' randomForest (random forest)
#' 
#' @param train_x, training set (features)
#' @param train_y, training labels
#' @param testset, testing set 
#' @param k, number of desired folds k, suggested #: 5
#' @param loss, custom loss function (accuracy_loss_fcn) 
#' that computes accuracy of model
#' 
#' @return a list of two vectors: one of CV accuracies 
#' and one of test accuracies both length k
#' 
#' @export
#' @examples
#'
#' # lda (linear discriminant analysis) example:
#'
#' cross_validation_lda <- CVgeneric(classifier = lda, train_x = binary_train, 
#' train_y = binary_train$expert_labels, testset = binary_test, 
#' k = 5, loss = accuracy_loss_fcn)
#'
#' >> $k_cv
#' [1] 0.8970160 0.8968033 0.8929218 0.8964096 0.8954358
#' $test_accuracy
#' [1] 0.9049573 0.9050712 0.9052991 0.9046154 0.9051852
#'

CVgeneric <- function(classifier, train_x, train_y, testset, k, loss = accuracy_loss_fcn){
  set.seed(999)
  k_accuracy <- NULL
  test_acc <- NULL
  folds <- createFolds(c(1:length(train_y)), k = k)
  for (i in 1:k){
    big_fold <- train_x[-folds[[i]], ]
    kth_fold <- train_x[folds[[i]], ]
    test_type <- classifier(expert_label~NDAI + SD+CORR, data = kth_fold[1:100, ])
    if (class(test_type) == "randomForest.formula"){
      model <- classifier(factor(expert_label) ~ NDAI + SD + CORR, data = big_fold, ntree = 300, nodesize = 19)
    } else{
      model <- classifier(expert_label~NDAI + SD + CORR, data = big_fold, family = "binomial")
    }
    
    if (class(model)[1] == "lda" | class(model)[1] == "qda"){
      preds <- predict(model, kth_fold)$class
      test_preds <- predict(model, testset)$class
    } else if (class(model)[1] == "glm") {
      probs <- predict(model, kth_fold)
      test_probs <- predict(model, testset)
      
      preds <- ifelse(probs > 0.5, 1, 0)
      test_preds <- ifelse(test_probs > 0.5, 1, 0)
    } else if (class(model)[1] == "randomForest.formula"){
      preds <- predict(model, kth_fold)
      test_preds <- predict(model, testset)
    }
    acc <- loss(preds, train_y[folds[[i]]])
    t_acc <- loss(test_preds, testset$expert_label)
    test_acc[i] <- t_acc
    k_accuracy[i] <- acc
  }
  result_list <- list(k_cv = k_accuracy, test_accuracy = test_acc)
  return(result_list)
}


