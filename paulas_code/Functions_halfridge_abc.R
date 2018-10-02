

######## 1) Half -ridge function ###########################################################

# penalty <- penalty[p]
# X <- x
# Y <- y 
# chain_length <- chain_length[v]

halfridge <- function(cue_validities, Predictors, X, Y, penalty, sigma_2,chain_length) 
{
    
    ## penalty contains 1/eta_2 so the inverse of eta_2, and grows from small to large!
    ## sigma_2 is fitted error variance (Linear regression residual) and since model is circular, we hold it constant and try out different values
       
      # assumption that cue directionalities are known in advance (Dawes, 1979) 
      unitweights <- sign(cue_validities)     
      col_pos <- (unitweights > 0)*1
      
      ### L2 half-ridge Fitting (Training Data)   ########
      # posterior mean is a vector that is Predictors-dimensional, as a function of penalty
      post_mean <- as.vector(ginv(t(X)%*%X + sigma_2*penalty*diag(Predictors)) %*% (t(X)%*%Y))

       # posterior variance is a PredictorsxPredictors covariance square matrix
      post_var <- sigma_2 * ginv(t(X)%*%X +  sigma_2*penalty*diag(Predictors))
      
      
     ## when penalty <- 0.0000001, the problem lies in sampling below not in solve() above!
      ## the error message is in rtmvnorm: sigma is sometimes not positive- definite. it depends on the training set
      ## there is a way of making it p-d
      # this problem only really occurs when training size is very small 
      
      # problem with penalty = 0, the OLS weights are often very negative which is more likely to not get truncated 
      # and all posterior samples are NaN -> have to run for longer
      # making the matrix post_var positive-definite
      if (is.symmetric.matrix(post_var)){
        if (!is.positive.definite(post_var)){
        
      PD <- nearPD(post_var) #ensureSymmetry = TRUE
      post_var <- PD$mat
      
        }
      }
      
     
     # ~N: or just sample from a multivariate Gaussian with above mean and variance and only keep positive samples
     samples1 <- mvrnorm(n = chain_length, mu = post_mean, Sigma =  post_var, tol = 1e-6)
     
    
     # Cut-off rule: reflects the correct cue directions that are known in advance (col_cues)
     # take the correct orthant
     samples1[,col_pos==1][samples1[,col_pos==1]<0]<-NaN
     samples1[,col_pos==0][samples1[,col_pos==0]>0]<-NaN
    
     #samples1[samples1 < 0] <- NaN
     post_weights <- colMeans(samples1, na.rm = TRUE)
     
     
     # this resclaing of means with eta (not eta_2!) now always works no matter how large penalty is
#       rescaled_weights <- post_weights/(sqrt(1/penalty))  #rescaled with eta. 1/penalty is eta_2 as 1/eta_2 = penalty
#    # cant get this variance to be identity matrix because its not of the posterior samples 
#      post_cov <- cor(samples1)
#      post_cov <- post_cov/(sqrt(1/penalty)) 
      
   
      # larger chain length does not help with convergence to 0.79 at penalty 1e+06
#       samples <- as.data.frame(rtmvnorm(n= chain_length, mean = post_mean,sigma = post_var, 
#                           lower= rep(0, length(post_mean)), upper=rep(Inf, length = length(post_mean)),
#                           algorithm = "gibbs", burn.in.samples = 1000, thinning = 10))
#                     
      
        cat(paste("\n\nNumber of Na samples  = ", colSums(is.na(samples1))[1]), "\n")    
      
      

        print(post_weights)
  
  
        # if all samples are NAn
        i = 0
        while (any(is.na(post_weights)) & i < 3){
             
          i = i +1 # after 4 iterations it stops
          # sometimes the post_mean is so fucked up it will always give all Nan samples
          samples1 <- mvrnorm(n = 50000, mu = post_mean, Sigma =  post_var, tol = 1e-6)
          
          samples1[,col_pos==1][samples1[,col_pos==1]<0]<-NaN
          samples1[,col_pos==0][samples1[,col_pos==0]>0]<-NaN
          
          cat(paste("\n\nNumber of Na samples  = ", colSums(is.na(samples1))[1]))    
          
             #####    POSTERIOR MAXIMUM: MEAN OF POST. DISTRIBUTION          
          # eta = solve(sqrt(penalty[p])) re-scale or not?
          post_weights <- colMeans(samples1, na.rm = TRUE)
          
        }

#         if (any(is.na(post_weights))){
#           stop('NA error')
#         }
  

  return(post_weights)
}



## Heatmap function


# penalty <- penalty[p]
# X <- x
# Y <- y 
# chain_length <- chain_length[v]

heatmaps <- function(Predictors, X, Y, penalty, sigma_2,chain_length) 
{
  
  penalty <- 0.00000001
  penalty <- 1000000
  
  ### L2 half-ridge Fitting (Training Data)   ########
  # posterior mean is a vector that is Predictors-dimensional, as a function of penalty
  post_mean <- as.vector(ginv(t(X)%*%X + sigma_2*penalty*diag(Predictors)) %*% (t(X)%*%Y))
  
  
  # posterior variance is a PredictorsxPredictors covariance square matrix
  # Problem: with penalty = 0 this is just OLS with negative weights possible
  post_var <- sigma_2 * ginv(t(X)%*%X +  sigma_2*penalty*diag(Predictors))
  
  # calculate the first and second moments: mean and covariance matrix with the above
  # at penalty 0 doesnt work otherwise
  PD <- nearPD(post_var) #ensureSymmetry = TRUE
  post_var <- PD$mat
  
  moments <- mtmvnorm(mean = post_mean, sigma = post_var, 
           lower = rep(0, length(post_mean)),
           upper = rep(Inf, length = length(post_mean)),
           doComputeVariance=TRUE,
           pmvnorm.algorithm=GenzBretz())
  
  moments$tmean # all positive so truncation worked
  # this is the variance we want to plot
  covariance_mat <- moments$tvar
  
  # penalty near 0
  rescaled <- covariance_mat/((1/penalty))  #rescaled with eta2 as I didnt use eta. 1/penalty is eta_2 as 1/eta_2 = penalty
  
  
  #near inf
  covariance_mat <- round(covariance_mat,7)
  rescaled <- covariance_mat/(sqrt(1/penalty))  #rescaled with eta2 as I didnt use eta. 1/penalty is eta_2 as 1/eta_2 = penalty
  

  # heatmap of both small and large penalty paramter:
  
  x11(width=20, height=15) 
  

  pdf("cov_matrix_penalty_0.pdf", width = 15, height = 10)
  heatmap.2(rescaled, Rowv = FALSE, Colv = FALSE, trace = "both", labRow = 
              c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"), 
            labCol = c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"),
            srtRow = 10, srtCol = 10, col = "gray.colors")
   
  dev.off()
  
  
  pdf("cov_matrix_penalty_inf.pdf", width = 15, height = 10)
  # at penalty = 1000000000 1e+11
  heatmap.2(rescaled1, Rowv = FALSE, Colv = FALSE, trace = "both", labRow = 
              c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"), 
            labCol = c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"),
            srtRow = 10, srtCol = 10, col = "gray.colors")
  
  
  dev.off()

  
  
  # this resclaing of means with eta (not eta_2!) now always works no matter how large penalty is
  #       rescaled_weights <- post_weights/(sqrt(1/penalty))  #rescaled with eta. 1/penalty is eta_2 as 1/eta_2 = penalty
  #    # cant get this variance to be identity matrix because its not of the posterior samples 
  #      post_cov <- cor(samples1)
  #      post_cov <- post_cov/(sqrt(1/penalty)) 
  
 
  # larger chain length does not help with convergence to 0.79 at penalty 1e+06
  #       samples <- as.data.frame(rtmvnorm(n= chain_length, mean = post_mean,sigma = post_var, 
  #                           lower= rep(0, length(post_mean)), upper=rep(Inf, length = length(post_mean)),
  #                           algorithm = "gibbs", burn.in.samples = 1000, thinning = 10))
  #                     
  
  cat(paste("\n\nNumber of Na samples  = ", colSums(is.na(samples1))[1]), "\n")    
  
  

}







ridge.predictions <- function(dataset, cv, y.pos,Predictors, sigma_2, penalty) 
{
  
  ## penalty contains 1/eta_2 so the inverse of eta_2, and grows from small to large!
  ## sigma_2 is error variance and is estimated from sd of noise in generating data
  
  
  attach(dataset)
  labels <- names(dataset)
  if(!exists(labels[y.pos])){
    stop('the dataset does not contain the binary dependent variable!')
  }
  
  #  number k of different test set seprations for the loop below
  
  k <- nrow(cv) 
  
  
  #sigma_2 <-  var_error
  
  
  ##   initiate empty lists
  pred.posterior <- vector('list', k)
  
  
  for (i in 1:k){
    
    
    if (i == k/10) cat(paste("\n\nSIMULATION COMPLETE  = 10%  , Time: ",Sys.time() ))
    else if (i == k/4) cat(paste("\n\nSIMULATION COMPLETE  = 25%  , Time: ",Sys.time() ))
    else if (i == k/2) cat(paste("\n\nSIMULATION COMPLETE  = 50%  , Time: ",Sys.time() ))
    else if (i == k - k/4) cat(paste("\n\nSIMULATION COMPLETE  = 75%  , Time: ",Sys.time() ))
    else if (i == k - k/10) cat(paste("\n\nSIMULATION COMPLETE  = 90%  , Time: ",Sys.time() ))
    
    
    testset <- dataset[cv[i,]==0, ] # is data frame
    trainset <- dataset[cv[i,]==1, ]
    
    
    # Design matrix X (bdata_diff)
    X <- as.matrix(trainset[-y.pos])
    
    # dependent variable Y (vector)
    Y <- as.matrix(trainset[y.pos])      
    
    
    ## Location in the loop to the prompt
    cat(paste("\n\nTEST SET NUMBER  = ", i))
    
    
    post_predict <- matrix(nrow=length(penalty),ncol=nrow(testset))          
    
    for (p in 1: length(penalty)){
      
      
      ### Ridge Regression Fitting (Training Data)   ########
      
      ## assuming linear regression, not logistic regression!
      
      # posterior mean is a vector that is Predictors-dimensional, as a function of penalty
      post_mean <- as.vector((solve(t(X)%*%X + ((sigma_2 * penalty[p])*diag(Predictors))) %*% (t(X)%*% Y)))
      # posterior variance is a PredictorsxPredictors covariance square matrix
      post_var <- sigma_2 * solve(t(X)%*%X + ((sigma_2 * penalty[p])*diag(Predictors))) 
      
      # linear regression case
      # post_mean <- as.vector(solve(t(X)%*%X) %*% (t(X)%*% Y)) 
      
      #n x Predictors - samples
      samples <- rtmvnorm(n=20000,mean = post_mean,sigma = post_var, 
                          lower= rep(0, length(post_mean)), algorithm = "gibbs")
      # or   algorithm="gibbs", burn.in.samples=100
      
      
      #####    POSTERIOR MAXIMUM: MEAN OF POST. DISTRIBUTION          
      
      post_weights <- colMeans(samples) # posterior mean is used as weight estimates
      
      ## never rescaled them by eta for prediction!
      
      
      # check convergence to ~ 0.79 (sqrt(2/pi))
      cat(paste("Weights at penalty (1/eta_2) = ", penalty[p]))
      #print(post_weights)   
      
      # weights devided by eta w/eta converge to sqrt (2/pi)
      if (p > 1) # because penalty[1] = 0
        print(post_weights/solve(sqrt(penalty[p])))  
      
      # for the case without sigma_2 in calculation!
      #print(post_weights/(sqrt(sigma_2/penalty[p])))  
      
      
      ### Ridge Regression Prediction (Test Data)   ############################   
      
      testdata <- as.matrix(testset[ ,-y.pos])
      
      # Matrix multiplication with testset, and take the sign to see whether A or B is predicted
      
      post_predict[p, ] <-   sign(testdata %*% post_weights)
      
      #       
      #       plot(density(testdata %*% post_weights), main = "Posterior predicted data (xw)
      #        (before thresholding)")
      
      
      # making predictions to be either y = 0 or 1
      post_predict[p, ] <- (post_predict[p, ]==1)*1
      
    }
    
    
    pred.posterior[[i]] <- post_predict   
    
  }
  
  return(pred.posterior)
}






















#####    Covariance Function     ########################################
##
##
##
##
covariance <- function(level, Predictors, noise)
{
  
  
  if (level == 1) { cov.level<- 0
    } else if (level== 2) {cov.level<-.1
    } else if (level == 3) { cov.level<- .2                        
    } else if (level == 4) { cov.level<- .3                              
    } else if (level == 5) { cov.level<-.4                           
    } else if (level == 6){ cov.level<-.5                            
    } else if (level == 7){ cov.level<-.6                                                             
    } else if (level == 8){ cov.level<-.7  
    } else if (level == 9){ cov.level<-.8 
    } else if (level == 10){ cov.level<-.9 
    } else if (level == 11){ cov.level<-1.0 
    }
                          
  covmat<-matrix(1, Predictors, Predictors)
  
  covmat[upper.tri(covmat)]<-cov.level
  
  covmat<-covmat+ matrix(rnorm(Predictors^2, 0 , noise), Predictors, Predictors)
  
  covmat[!upper.tri(covmat)]<-0
  
  covmat<-covmat+t(covmat)
  
  diag(covmat)<-1
  
  
  return(covmat)
  
}





## 	        3.) Function cv.indexing
##
##	cross-validation partitioning for current environment dataset,
##	takes seed as input to be able to reproduce results,
##	However, can use different seeds here to get different cross 
##	validation results for the same data set 

## Generates 'k' sets of indices, where each set represents a split
## of a data set of size 'N' into two disjoint data sets (sometimes called a
## 'folder').
##  
## Within each split, part of the data is called a 'test' data set, and the
## remaining data points form the 'training' data. None of the 'test' sets,
## overlap, and their union corresponds to the complete data set.
##
## This function returns a k-by-n binary matrix 'cv'. Each row in 'cv' 
## represents a split. If cv[i, j] == 1, then data point 'j' is in the 
## training set of partition 'i'. If cv[i, j] == 0, then data point 'j' is 
## in the test set of partition 'i'.
##
## The number of test points in the last partition will be larger than in 
## the other partitions if 'N' is not a multiple of 'k'.

cv.indexing <- function(k, N, percent)
{
  ## 	Arguments:
  ##	k 	number of partitions (set above)
  ##	N	size of the current data set (set above)
  ##	seed 	seed used for random partitions (set above) 
  
  #   if (k >= N)
  #     stop('Number of partitions should be less than number of data points!')   
  #   
  
  #set.seed(seed) # For reproducibility of results, only
  # dont use it now for the perm function!
  
  test.set<- round(percent * N)
  # the size of  a test set is 20% of N
  
  
  cv <- matrix(rep(1, k * N), nrow = k) # Allocate output matrix
  
  
  for (i in 1:k){
    
    # generate new permuation for each row i 
    perm <- sample(1:N, N)
    
    # define the test set items from columns in perm, to be marked by 0 
    # while the training set items stay 1
    cv[i,perm[1:test.set]] <- 0
    
    
  }
  
  return(cv)                              
}

    ######## 2) Truncated Linear Regression function ###########################################################
    
    # penalty <- penalty[p]
    # X <- x
    # Y <- y 
    
    truncated.linear <- function(Predictors, X, Y, penalty, test,sigma_2) 
    {
      
      ## penalty contains 1/eta_2 so the inverse of eta_2, and grows from small to large!
      ## sigma_2 is fitted error variance (Linear regression residual) and since model is circular, we hold it constant and try out different values
      
      penalty <- 0
      
      ### L2 half-ridge Fitting (Training Data)   ########
      # posterior mean is a vector that is Predictors-dimensional, as a function of penalty
      post_mean <- as.vector(ginv(t(X)%*%X + sigma_2*penalty*diag(Predictors)) %*% (t(X)%*%Y))
      
      # or sample from re-scaled weights directly?
      
      # posterior variance is a PredictorsxPredictors covariance square matrix
      post_var <- sigma_2 * ginv(t(X)%*%X +  sigma_2*penalty*diag(Predictors))
      
      
      
      # cant draw from truncated Normal where post_mean are produced with 0 penalty
      samples <- rtmvnorm(n=10000,mean = post_mean,sigma = post_var, 
                          lower= rep(0, length(post_mean)), upper=rep(Inf, length = length(post_mean)),
                          algorithm = "gibbs", burn.in.samples = 1000, thinning = 10)
      
      # rejection sampling is more exact. inefficient when support area is small (not here)
      
      # or   algorithm="gibbs", burn.in.samples=100
      
      
      #####    POSTERIOR MAXIMUM: MEAN OF POST. DISTRIBUTION          
      # eta = solve(sqrt(penalty[p])) re-scale or not?
      post_weights <- colMeans(samples)  #solve(sqrt(penalty)) = Inf. so that penalty = 0 # posterior mean is used as weight estimates
      
      output <- test  %*%  post_weights   
      LR_output <- as.vector(sign(output))  #NxPredictors %*% Predictorsx1  = Nx1
      
      # check convergence to ~ 0.79
      #       cat(paste("Weights at penalty (1/eta_2) = ", penalty[p]))
      #       #print(post_weights)   
      #       
      #       # weights devided by eta w/eta converge to sqrt (2/pi)
      #       if (p > 1) # because penalty[1] = 0
      #         print(post_weights/solve(sqrt(penalty[p])))  
      #       
      
      
      return(post_weights)
    }

################# 3a) Ginv-inverse: Linear Regression Function ############################################
##
# y.pos <- ncol(dataset)

Linear.regression <- function(x, y, test, Predictors, labels) 
{
  

##---------------------- Fitting --------------
      X <- x
      Y <- y 
    
     
      
      coef_LR <- ginv(t(X)%*%X) %*% (t(X)%*%Y)
# not solvable for lots of redundancies, the same reason that coefficients are NA usually below in fit

# needs to be exactly consistent with how posterior outcomes matrix is computed 
      coef_LR <- round(coef_LR,4)

# Just needs to be rounded as well in order to achieve 100% agreement at penalty=0 with COR,
# because the matrix multiplication test  %*%  coef_LR does not reduce to zero but to -4.440892e-16 somehow in R. that messes up signs.


###------------------- Prediction -------------------------------------  

      # Instead of rounding the output matrix, the coefficients above are rounded to 4 decimals like w. That makes output = outcomes (any column)
      output <- test  %*%  coef_LR    
      LR_output <- as.vector(sign(output))  #NxPredictors %*% Predictorsx1  = Nx1
      

##----------------------- guessing for now not allowed (PNAS paper) ----------------------------------------------------------------------
#try out with guessing
#     for (j in 1:length(LR_output)){  
#       if (LR_output[j]==0) LR_output[j] <- sign(rnorm(1))  
#       }
#         
#     pred.linear[[i]] <- LR_output  # no guessing
#        
#   }
  
  return(LR_output)   
  
}





#############       4.) Regression Prediction Graph (accuracies)########################

regression.graph <- function(predictions.regr, test.labels)
{
  
  #predictions.regr <- predictions.regression[[1]]
  
  # predictions.regression      is a vector from a list containing the 0/1 predictions
  # test.labels                 is a vector of the correct outcomes from the current testset
  
  n <- length(predictions.regr)
  m <- length(test.labels)
  all.equal(n,m)
  
  prop_regr <- sum(test.labels==predictions.regr)/m
  
  
  return(prop_regr)
  
}









#####     6.) Pred.Graph Function (Ridge accuracies)    #########################
##
##
##
pred.graph <- function(predictions, test.labels){
  
  
  # predictions <- predictions.regridge[[i]]
  
  # predictions is matrix with penalty x testpredictions
  
  
  # 	get the number of test predictions 
  n <- ncol(predictions)
  
  
  # Thresholding the predictions first:
  # can vary the threshold point!
  #       thres <- 0.5
  #       
  #       predictions <- (predictions > thres)*1 
  #       
  
  
  
  raw.acc <- vector("numeric",nrow(predictions))
  
  for (l in 1:nrow(predictions)){
    
    
    raw.acc[l] <- sum(predictions[l, ] == test.labels)/n
    
    
  }
  
  # put the vector raw_acc into a list each time, and return it	
  
  propList <- list(raw.acc=raw.acc)
  return(propList)
  
}





########## 9.) Tallying Learning Function ##############################
##

#y.pos <- ncol(dataset)

tallying.learning <- function(trainset, test, Predictors)
{ 
#   


    cue_validities_raw <- vector('numeric', Predictors) 
    cue_validities <- vector('numeric', Predictors)
    for (c in 1:Predictors){
      # estimate the ecological cue validity of each cue as v = R/(R+W)
      if (sum(trainset[,c]==trainset[ ,ncol(trainset)]) == 0) { cue_validities[c] <- 0 
      } else  
        cue_validities_raw[c] <- sum(trainset[,c]==trainset[ ,ncol(trainset)])/(sum(trainset[,c]==1)+sum(trainset[,c]==-1))              
        cue_validities[c] <- cue_validities_raw[c] - 0.50    
    }
      
    # the validities - 0.50 are automatically same sign as regression betas 
    unitweights <- sign(round(cue_validities,4))      # make sure they are ROUNDED to 4 DECIMALS AS WELL.
    
    
  ############ Prediction Tallying (Testdata) ###########################################    
   
    # matrix multiplication of unit weights and testdata, if > 0, then +1 (A), if < 0, then B(-1)
    tl_predict <- sign(test %*% unitweights)

# THE 0 CASES HAVE TO BE DELETED FROM OTHER MODELS TOO!
    ## if the sum of all cue differences is still 0, then guess, different each time!!
#     for (j in 1:length(tl_predict)){  
#       if ( tl_predict[j]==0)
#       {
#         tl_predict[j] <- sign(rnorm(1))
#       }
#     }   ## all +1 and -1 now!

  
  return(tl_predict)

}

###     8.)  Truncated Tallying  Function        #################################
##
## Tallying directed: estimates valence of unit weights from the whole dataset


truncated.tallying <- function(cue_validities, trainset, test, Predictors) 
{
  
  # assuming cue directionalities are known from the whole data set already
  unitweights <- sign(cue_validities)     
  
# only positive (incorrect) weights
#  unitweights <- rep(1,Predictors)   
  

  # matrix multiplication of unit weights and testdata, if > 0, then +1 (A), if < 0, then B(-1)
  tl_predict <- sign(test %*% unitweights)
  
  # THE 0 CASES HAVE TO BE DELETED FROM OTHER MODELS TOO!
#   ## if the sum of all cue differences is still 0, then guess, different each time!!
#       for (j in 1:length(tl_predict)){  
#         if ( tl_predict[j]==0)
#         {
#           tl_predict[j] <- sign(rnorm(1))
#         }
#       }   
  
  
  return(tl_predict)
  
}




#####         9.) Tallying.graph (accuracies) Function    ######################################
##
###
##
##

tallying.graph <- function(predictions.tallying, test.labels)
{
  
  ## function provides a summary of the performance (accuracy) of the tallying 
  ## heuristic
  
  # get the number of test predictions/length of predictions vector
  
  
  n <- length(predictions.tallying)
  m <- length(test.labels)
  all.equal(n,m)
  
  
  
  prop <- sum(test.labels==predictions.tallying)/m
  
  # prop contains accuracies
  
  return(prop) # a single number 
  
}




#### 10. TTB Learning Function ##########################################################################


#trainset <- dataset

ttb.predictions <- function(trainset, test,  Predictors)
{
 
  
  # trainset[ ,ncol(trainset)] still has y variable in it
    
  
    cue_validities_raw <- vector('numeric', Predictors) 
    cue_validities <- vector('numeric', Predictors)
    for (c in 1:Predictors){
      # estimate the ecological cue validity of each cue as v = R/(R+W)
      if (sum(trainset[,c]==trainset[ ,ncol(trainset)]) == 0) { cue_validities_raw[c] <- 0 
      } else  
        cue_validities_raw[c] <- sum(trainset[,c]==trainset[ ,ncol(trainset)])/(sum(trainset[,c]==1)+sum(trainset[,c]==-1))              
      cue_validities[c] <- cue_validities_raw[c] - 0.50    
    }
 
    cue_validities <- round(cue_validities, 4)
    

    # ROUND TO 2 DECIMALS to make EQUAL TO DIAGONAL ORDER: Trying out 4 now to take care of rounding order problem (agreement is still not 1 sometimes)
    # needs to be 4
    cue_order <- order(abs(cue_validities), decreasing = TRUE)
    ## if two are the same, order chooses the first element, i.e., the one earlier in the vector.
  
  
    ############ PREDICTING TESTDATA WITH TTB (PREDICTION) ###########################################      
  
    ttb_predict <- vector("numeric",nrow(test))   
    for(r in 1:nrow(test)){
      v <- 1
      # circulates through elements 1:5 of cue_order in order of abs. vailidity 
      while (test[r,cue_order[v]]== 0){            
        
        if (v == length(cue_order)){ 
          ttb_predict[r] <- 0 #sign(rnorm(1)) #0 if no cue discriminates, guess between A(+1) and B(-1) 
          break  
        }       
        v <- v + 1  # go to the next element in the cue_order
        
      } # while loop only breaks when test ~= 0 , and then the loop is halted completely!!!
      
      if (test[r,cue_order[v]]== 1 & cue_validities[cue_order[v]] > 0){           
        ttb_predict[r] <- 1  # prediction is sign
        
      } else if (test[r,cue_order[v]]== 1 & cue_validities[cue_order[v]] < 0){ 
        ttb_predict[r] <- -1  # 
        
      } else if (test[r,cue_order[v]]== -1 & cue_validities[cue_order[v]] > 0){
        ttb_predict[r] <- -1 #
        
      } else if (test[r,cue_order[v]]== -1 &  cue_validities[cue_order[v]] < 0){
        ttb_predict[r] <- 1  #           
      }
    } # end of r loop

   
  return(ttb_predict) # return the whole list with 20 elements
  
} # end of function



######  11.) TTB graph function (proportion correct) ######################
##
##
ttb.graph <- function(predictions.ttb, test.labels)
{
  
  ## function provides a summary of the performance (accuracy) of the tallying 
  ## heuristic
  ## input 'predictions.tbb' is always only 1 element of the list (vector)
  
  #predictions.ttb <- predictions.ttb[[i]]
  
  n <- length(predictions.ttb)
  m <- length(test.labels)
  #all.equal(n,m)
  
  
  prop.ttb <- sum(test.labels==predictions.ttb)/m
  
  # prop contains accuracies
  
  return(prop.ttb) # a single number 
  
  
}


