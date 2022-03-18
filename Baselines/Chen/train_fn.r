naive_chen_train <- function(dt_train){
	library(SVMW)
	index = which(dt_train$y_obs_train > quantile(dt_train$y_obs_train,0.6))
	x_names = -which(names(dt_train) %in% c('t_obs_train','t_opt_train', 'y_obs_train','t_dens_obs_train'))
	
	### We use the penalized quantile regression to enhance the model fitting.
	model  = svm(x = dt_train[index,x_names], y = dt_train$t_obs_train[index], w= (dt_train$y_obs_train[index]/dt_train$t_dens_obs_train[index]), type="eps-regression",
		epsilon = 0.15, scale=FALSE)
	pred_train<-predict(model,dt_train[,x_names])
	return(data.frame(pred_train))
}
naive_chen_test <- function(dt_train, dt_test){
	library(SVMW)

	index = which(dt_train$y_obs_train > quantile(dt_train$y_obs_train,0.6))
	x_names = -which(names(dt_train) %in% c('t_obs_train','t_opt_train', 'y_obs_train','t_dens_obs_train'))
	
	### We use the penalized quantile regression to enhance the model fitting.
	model  = svm(x = dt_train[index,x_names], y = dt_train$t_obs_train[index], w= (dt_train$y_obs_train[index]/dt_train$t_dens_obs_train[index]), type="eps-regression",
		epsilon = 0.15, scale=FALSE)
	pred_test<-predict(model,dt_test[,x_names])
	return(data.frame(pred_test))
}

############### propensity score part
#mydata = data.frame(T = dt_train$y_obs_train,X = dt_train[,x_names])
#model.num = lm(dt_train$t_obs_train~1,data = dt_train[,x_names])
#ps.num= dnorm((dt_train$t_obs_train-model.num$fitted)/(summary(model.num))$sigma,0,1)
#model.den = gbm(dt_train$t_obs_train~.,data = dt_train, shrinkage = 0.0005, interaction.depth = 4, distribution = "gaussian",n.trees = 20000)
#opt = optimize(F.aac.iter,interval = c(1,20000), data = dt_train, ps.model = model.den, ps.num = ps.num,rep = 50,criterion = "pearson")

#best.aac.iter = opt$minimum
#best.aac = opt$objective

# Calculate the inverse probability weights
#model.den$fitted = predict(model.den,newdata = dt_train,n.trees = floor(best.aac.iter), type = "response")
#ps.den = dnorm((dt_train$t_obs_train-model.den$fitted)/sd(dt_train$t_obs_train-model.den$fitted),0,1)
#weight.gbm = ps.num/ps.den


