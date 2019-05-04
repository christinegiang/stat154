# stat154

We used two methods of splitting the data, both of which accounted for the important assumption that the image data is non-i.i.d. In Method 1, we divided the data into grids such that there are ten grids on the y-coordinate axis and ten grids on the x-coordinate axis, producing a total of 100 grids. We randomly sampled 80% of the grids for our training data, 10% for our validation data, and the remaining 10% for our test data. By sampling grids rather than individual pixels, we account for the fact that the pixels are non-i.i.d. In Method 2, we arbitrarily assigned each of the three images to our training, validation, and test sets, which also helped preserve our non-i.i.d. assumption. 

Next, we selected the three features that we identified as the most significant. Based on correlation with the expert labels and non-collinearity, we selected NDAI, SD, and CORR as our features. This selection decision was enforced by our scatter plots, in which we color coded the values of the three features for each pixel by their expert label. We observed distinct separation of the three colors, or expert labels, for these three features. We ruled out the five radiance angles due to their collinearity with NDAI. Furthermore, we eliminated the x and y-coordinates under our assumption that geographical location does not affect cloud presence. 

We used four models to classify the pixels as cloud or non-cloud, using the expert labels provided in the three images to train and test our models. We used LDA, QDA, logistic regression, and random forest. To evaluate model performance, we obtained cross validation and test accuracies. We combined our training and validation data from Method 1 and then applied 5-fold cross validation, where we used 4 folds to train the model and the hold out set as our validation data. Rather than obtaining average CV accuracy, we extracted the CV accuracies across each fold to determine any trends across the folds, for which none were observed. We ran each of the models trained on the k-1 folds on our test set to obtain test accuracies, which helped us to identify random forest as our best-performing model. Beyond CV and test accuracies, we compared ROC curves, Youden points, AUC values, and confusion matrices for each of the four methods. The Youden point was used to identify the maximum difference between the true positive and false positive rates. 

After selecting random forest as our classification model of choice, we ran diagnostic plots to further evaluate this choice of model. We used PCA transformation in two different ways. In the first method, we transformed the three selected raw features and pulled out the top two principal components, which accounted for roughly 89 percent of the variation. We used 85 percent as our threshold. We did not notice any significant improvement in model performance across LDA, QDA, logistic regression, and random forest as evidenced by accuracy or AUC under PCA transformation. In our second PCA transformation method, we expanded our feature selection to include the five radiance angles, and used the first three PCs for greater variation capture. Accuracy results from this second PCA transformation were very similar to those of the first PCA transformation using only NDAI, SD, and CORR. This confirmed our original assumption that the information of the five radiance angles is broadly captured by NDAI. 

We assessed stability in Method 2 of splitting the data by re-assigning each of the three images to our training, validation, and test sets. Model performance did not change noticeably with this permutation, confirming good model stability. Overall, we determined that random forest is an appropriate classification model for the cloud image data set. 
