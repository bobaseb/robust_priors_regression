    ##  Exact Bayesian Solution in the end 
    

## preamble 

    rm(list=ls())

    packages <- c('clusterGeneration', 'MASS', 'ggplot2', 'lars', 'psych', 'matrixcalc', 'abind', 'lattice', 'coda', 
                  'foreach', 'doParallel')
    lapply(packages, require, character.only = TRUE)
    
    registerDoParallel(cores= 8)   
    # 8 at work, 12 on server
    
    source("Functions_exactBayesian_abc.R")
    
 ## -------------------------------------------------------------------------------------------------------------------    
      
    ALL_DATA <- c("house.world","mortality","cit.world","prf.world","bodyfat.world", "car.world","cloud",
                "dropout","fat.world", "fuel.world", "glps",
                "homeless.world", "landrent.world", "mammal.world", "oxidants",
                 "attractiveness.men", "attractiveness.women", "fish.fertility","oxygen", "ozone")
 
    
    all_results <- vector("list", 20)
    
    #Run a parallel loop, each iteration should be a different job you need doing (e.g. fitting your model with different starting parameters)
    # all_results<-foreach(e=1:20) %dopar%
    # {
    #  
       
    
        # inside dopar
      packages <- c('clusterGeneration', 'MASS', 'ggplot2', 'lars', 'psych', 'matrixcalc', 'abind', 'lattice', 'coda')
      lapply(packages, require, character.only = TRUE)
      
      source("Functions_exactBayesian_abc.R")  
    
    
    for (e in 1:20){
    
    ##-------------------------- Imread datasets  ---------------------------------------------------------
    dataset <- read.table(paste(ALL_DATA[e],".txt", sep = ""), header = TRUE) # no space allowed
    
    #for oxygen exclude 4 predictors
    # if (ALL_DATA[e] == "oxygen"){
    # dataset <- dataset[    ,-c(7,9,10)]  #without 4 redundant predictors
    # }
    # 
    # # for ozone get rid of 1 perfectly correlated predictor
    # if(ALL_DATA[e] == "ozone"){
    # dataset <- dataset[ ,-5]
    # }
    # 
    # # oxidants without Wind: otherwise N=115 does not work
    # if (ALL_DATA[e] == "oxidants"){dataset <- dataset[ ,-5]}
    # 
     
    penalty <- c(1000000, 100000, 1000, 700.00000000, 330.07641174, 156.81303712,  74.49889703,  35.39301171,  16.81454797, 7.98827254,
                 3.79507663,  1.80296886, 0.85655628,  0.40693363,   0.19332644,   0.09184572,  0.03, 0.01, 0.001, 0.0001, 0.00001)
    
    ## CHECK whether DV is in column 3 for every dataset!
    y.pos <- 3 
    
    
    # make all datasets to contain cues in columns 5 - end
    col_cues <- c(5:ncol(dataset))
    
    # attach to be able to access headers
    labels <- names(dataset)
    labels <- labels[col_cues] # only the cues now
    
    ## assess the number of predictor X's, from column 5 to the end
    Predictors <- length(5:ncol(dataset))
    
    # number of objects
    N <- nrow(dataset)
    
    ## Choose the number of partitions for cross-validation
    k <- 1000 # test run with 5 to see if it works # larger but also less training set size  (5)
    
    # -----------------------------------------------------------------------------------------------------
    ## BEFORE creating paired_data: average redundancy between cues
    cov <- as.matrix(dataset[,col_cues])
    cor_mat <- cor(cov,method = "pearson")
    # the lower triangle of the cov matrix contains pairwise correlations
    min_cor <- min(cor_mat[lower.tri(cor_mat)])
    max_cor <- max(cor_mat[lower.tri(cor_mat)])
    abs_mean_cor <- mean(abs(cor_mat[lower.tri(cor_mat)]))
    # -------------------------------------------------------------------------------------------------
    
    ## makes no sense to estimate min_cor on different scale than mean(cor), either both absolute or not. I think try both.
    ## also makes way more sense that performance of strategies would depend on covariances on the level of the training data 
    ## not the raw data before forming comparisons. 
    
    
    
    ## ----------- Create Paired Data (ALL binary comparisons of objects) -----------------------------------------
    # take all possible 2-element combinations of N rows, and re-shuffle each combination by row to randomise order
    ## that way the dependent=y contains both 0's and 1's -> variation for correlation
    comb <- combn(N,m=2, FUN= function(x, ...) x[sample(length(x))] ) 
    y <- vector('numeric',ncol(comb)) # classification; A(+1) or B(-1)
    difference <- vector('numeric',ncol(comb))
    bdata_diff <- matrix(nrow=ncol(comb), ncol=length(col_cues))    
    for (i in 1:ncol(comb)){
      # takes out only the 2 rows from dataset that are compared at step i 
      binary <- dataset[comb[ ,i], ] # data.frame
      
      
      if (i == 1) {comparisons <- binary
      }else { comparisons <- rbind(comparisons,binary)
      }
      ## if          A(1)          >       B(-1) 
      ## always compare row 1 with row 2 (no matter which ones has the higher criterion value) upper row - lower row
      if (binary[1,y.pos] > binary[2,y.pos]){
        y[i] <- 1 #(A)
      } else y[i] <- -1 #(B)
      
      ## cue values (row 1) - cue values (row 2) 
      bdata_diff[i, ] <- as.matrix(binary[1,col_cues] - binary[2,col_cues]) #     
      #difference[i] <- (binary[1,y.pos] - binary[2,y.pos]) # this difference is always positive? 
    }    
    bdata_diff <- as.data.frame(bdata_diff)
    names(bdata_diff) <- labels# give it the cue labels
    ## combine cue data and dependent in a data frame
    paired_data <- data.frame(bdata_diff, dependent=y)
    #--------------------------------------------------------------------------------------------------------
    dataset <- paired_data # is the new dataset now for below
    
    
    # ------- Assess paired_data cue validities and order as v= R/R+W  -----------------
    cue_validities_raw <- vector('numeric', Predictors)  
    cue_validities <- vector('numeric', Predictors)  
    
    for (c in 1:Predictors){
      if (sum(paired_data[,c]==paired_data[ ,ncol(paired_data)]) == 0) { cue_validities[c] <- 0 # stays 0 now if it was 0 
      } else  
        cue_validities_raw[c] <- sum(paired_data[,c]==paired_data[ ,ncol(paired_data)])/(sum(paired_data[,c]==1)+sum(paired_data[,c]==-1)) 
      cue_validities[c] <- cue_validities_raw[c] - 0.5# back to same scale as regression weights as otherwise order can be different!
    } 
    cue_order <- order(abs(cue_validities), decreasing = TRUE) # values are between 0-1 anyway.
    
     
    
    # number of objects after evening out
    N <- nrow(dataset)
    
    ## --------------- Multivariate Error Term  ----------------------------
    xItems <- as.matrix(dataset[ ,1:Predictors]) 
    yLabelMat <- replicate(Predictors, dataset$dependent) 
    ## -------------------- Regression coefficients at penalty = 0, including covariance ------------------------------------------
    lm_multivariate <- lm.fit(xItems,yLabelMat) # xItems is already WITHOUT INTERCEPT (no column of1s)
    betas <- lm_multivariate$coefficients[ ,1] # coefficients are naturally in the columns
    betas_LR <- t(replicate(length(penalty), betas))
    
    
    ##------------------- Individual coefficients (no covariance) at penalty = inf. ---------------------------  
    ind_coef <- vector('numeric', Predictors)
    for (c in 1:Predictors){
      # run through each predictor variable/no intercept
      fmla <- paste("dependent ~ ", paste(labels[c], collapse= "+"), paste(" -1"))
      single_weights <- lm(fmla, data = dataset)
      ind_coef[c] <- single_weights$coefficients #assign ind. weights to vector       
    }  
    ind_coeffs <- t(replicate(length(penalty), ind_coef))
    colnames(ind_coeffs) <- labels
    
     
    # usually these sizes unless small data set 
    #training_size <- c(5, 10, 20, 100, 110, 115 ,120)
    
    training_size <- c(10,20, round(0.9*N))
    
    
    # if(ALL_DATA[e] == "oxygen"){
    #   training_size[3] <- 72 # 20 test set
    # } else if (ALL_DATA[e] == "ozone"){
    #   training_size[3] <- 35 # 20 test set
    # } 
    
    # precentage
    training_size <- training_size/N
    
      
    for (v in 1:length(training_size)){
      
      percent_training  <-  training_size[v]
      # ---------  Generate the cross-validation partitions: -----------------------------------------------------------
      percent <-(1 - percent_training)  #### Hold the testset (distinct from random training set) 
      training_sample_size <- percent_training*N
      cv <- cv.indexing(k, nrow(dataset), percent) # is random, after that i refers to the same thing always  
      
      ## Another cv matrix for resampling when zero-variance cases arise below
      percent_training <- training_size[v] 
      percent <-(1 - percent_training)  
      #extra dimensions for when all 1000  options are used up
      cv2 <- cv.indexing(k+1000,nrow(dataset),percent)
      
      ###-------------------------------------------- 1) MODEL FITTING ---------------------------------------------------------------------  
        dec_class_acc <- matrix(nrow=length(penalty), ncol = k)
        dec_TTB_accs <- matrix(nrow=length(penalty), ncol = k)
        dec_tallying_accs <- matrix(nrow=length(penalty), ncol = k)
         
        A_TTB_BayesianTTB <- matrix(nrow=length(penalty), ncol = k)
        A_Tallying_BayesianTallying <- matrix(nrow=length(penalty), ncol = k)
        A_LR_BayesianTTB <- matrix(nrow=length(penalty), ncol = k)
        A_LR_BayesianTallying <- matrix(nrow=length(penalty), ncol = k)
         
        predictions.LR <- vector('list',k) 
        predictions.ttb <- vector('list',k) 
        predictions.tallying <- vector('list',k) 
      
        results.linear.regression <- vector('numeric', k)
        results.ttb <- vector('numeric', k)
        results.tallying <- vector('numeric', k)
        re <- vector('numeric', k) # resampling
        cues <- vector('numeric', k) 
        
      for (i in 1:k){
          
          trainset <- dataset[cv[i,]==1, ]   
          testset <- dataset[cv[i,]==0, ]   
          # re-boot the Predictors for each i, otherwsie carry-over errors
          Predictors <- ncol(trainset)-1 # - dependent
        
##----------- Re-shuffling zero variance cases (incompatible with COR model) ------------------------------------------------------ 
          re[i] <- 0
          cov_mat <- cor(trainset[ ,1:Predictors])
          ## NA cases = zero variance cases, get resampled now until one is found without any zero variance cases
          while (any(is.na(cov_mat))){
      
            trainset <- dataset[cv2[i+re[i],]==1, ] # re[i] = 0 at i=1
            testset <- dataset[cv2[i+re[i],]==0, ]       
            re[i] <- re[i] + 1 # when 0 before, becomes 1 etc  
            
            cov_mat <- cor(trainset[ ,1:Predictors])    
          }  
          

#------Throwing out redundant predictors from x for both OLS and COR model fitting:      
      # == 1 has problems, so > .9999 grabs all the ones
      if (any(cov_mat[lower.tri(cov_mat)]  > 0.99999999)){ # if there is at least 1 complete redundancy (TRUE) in the lower triangle,  
        eliminate <- which(cov_mat > 0.99999999, arr.ind = T)
        var_delete <- vector('numeric', nrow(eliminate))
        for (f in 1: nrow(eliminate)){
          # only take those that are not the matrix diagonal
          if (eliminate[f,1] != eliminate[f,2]){        
            var_delete[f] <- eliminate[f,1] # store the row number of that first variable  
          }
        }
        # only take each variable once, in order, and only as many as necessary to get rid of redundancy
        redundant <- sort(unique(var_delete[var_delete > 0]))
        m <- length(redundant)# how many redundancies there are overall
     # deleting m-1 of the redundant predictors still gets rid of all redundancies
        trainset <- trainset[ ,-redundant[1:(m-1)]]  # get 2 out of 3 deleted 
        testset <- testset[ ,-redundant[1:(m-1)]]
        Predictors <- ncol(trainset) - 1 # dependent
    
      } # if loop



         ## Location in the loop to the prompt
        cat(paste("\n\nIteration  = ", i, "\n\n", "Predictors: ", Predictors))     
        cues[i] <- Predictors     

        test <- as.matrix(testset[ ,-ncol(testset)]) 
        test.labels <- testset[ ,ncol(testset)]

        # this identifies all rows that have just 0's, real ties, and excludes them
        nonties <-  as.logical(rowSums(abs(test)) != 0)
  
        x <- as.matrix(trainset[ ,1:Predictors])    
        y <- trainset$dependent
        
           
        ###--- ------------------  Regular Linear Regression: MAKE SURE IT USES manual (ginv) computation as well!
        # using ginv inverse function 
        predictions.LR[[i]] <- Linear.regression(x, y, test, Predictors, labels) #     
        ###--- ------------------  Regular TTB Heuristic ------------------------------------##################################### 
        predictions.ttb[[i]] <- ttb.predictions(trainset, test, Predictors)
        ###--- ------------------  Regular Tallying Heuristic ------------------------------------##################################### 
        predictions.tallying[[i]] <- tallying.learning(trainset, test, Predictors)

        # get TTB Heuristic accuracy
        results.ttb[i] <- ttb.graph(predictions.ttb[[i]], test.labels)
        # get TTB Heuristic accuracy
        results.tallying[i] <- ttb.graph(predictions.tallying[[i]], test.labels)
        # get LR accuracy
        # predictions.LR[[i]] <- LR_output
        results.linear.regression[i] <- regression.graph(predictions.LR[[i]], test.labels)
         
  ###---------------------------- COR Model --------------------------------------------------------------------------------
  
          meanW <- vector('list',length(penalty)) 
          for (p in 1: length(penalty)){

            ## Thus the posterior is a multivariate Gaussian with mean/median/mode at
            w <- matrix(nrow = Predictors, ncol = Predictors)           
          # cycling through all copies j = set j
                for (j in 1: Predictors){              
                  lambda <- matrix(nrow = Predictors, ncol = Predictors)
                 # for all lambda_ii = lambda
                  diag(lambda) <- penalty[p] # penalize all off-diagonals
                  lambda[j,j] <- 0  # except the direct weight
                  # for all i != j lamda_ij = 0
                  lambda[lower.tri(lambda)] <- 0
                  lambda[upper.tri(lambda)] <- 0
                  # I put the weights for each case into the columnns? first part has dimensions of inverse square matrix = 15x15, second half is 15x1
                  w[ ,j] <- ginv(lambda + t(x)%*%x) %*% (t(x)%*%y)
                 
              } # j loop
          
            
          ##---------------------- PREDICTION: ---------------------------------------------      
            # Matrix multiplication with testset, and take the sign to see whether A or B is predicted
            outcomes <-   test %*%  w
            
            # new epsilon rule: checked it with output when most cells are supposed to be zero (penalty 1e+06)
            epsilon <- 0.0009   # 9e-05
            
            # this could be cleaner than rounding the weights, output-based strategy
            outcomes[abs(outcomes) < epsilon] <- 0
            
          # rounding is necessary in order to get 100% agreement with TTB, order can be different otherwise as equal weights in w are not really equal 
            outcomes <- round(outcomes, 3)

          ## COR Linear Model: Binary classification  decision rule
            dec_class <- sign(rowSums(outcomes))   # every row contains a test item with all 4 cues, outcomes contain weighted cues        
            ## Dec Tallying 
            dec_tallying <- sign(rowSums(sign(outcomes)))  # is automatically zero when all cues are zero (before matrix mulitplication)
            # include guessing
            for (t in 1:length(dec_tallying)){  
              if (dec_tallying[t]==0)
              {
                dec_tallying[t] <- sign(rnorm(1))
              }
            }   ## all +1 and -1 now!
            
            ## Dec TTB
            dec_TTB <- vector('numeric',nrow(outcomes))     
            for (l in 1:nrow(outcomes)){          
              
              if (identical(outcomes[l,],rep(0, Predictors))){
                dec_TTB[l] <- sign(rnorm(1)) #0        # TTB guesses
              }else {
                
                temp <- abs(outcomes[l, ]) # is like the matrix diagonal: save it so that rounding errors are not taken over
                position <- order(abs(temp), decreasing = TRUE) # gives the same as order(abs(cue_validities), decreasing = TRUE)
                ## position already has the max as first position because otucomes integrates that. if sth is zero then it wont
                ## be the first in the order: this should give same answer as cue validities
                              
                dec_TTB[l] <- sign(as.numeric(outcomes[l,position[1]])) # if there is more than one max, it chooses the first one (Earlier in vector)              
              }       
            }
          
        ## Test cases:
        ## dec_tallying == predictions.tallying[[i]]
        ## dec_TTB == predictions.ttb[[i]]

          ## Performance of the TTB decision rules      
          dec_TTB_accs[p,i] <- sum(dec_TTB == test.labels)/length(test.labels)
          dec_tallying_accs[p,i] <- sum(dec_tallying == test.labels) /length(test.labels)
          dec_class_acc[p,i] <- sum(dec_class == test.labels)/length(test.labels)
          
          A_TTB_BayesianTTB[p, i] <- sum(predictions.ttb[[i]] == dec_TTB)/length(test.labels) #is 1 when penalty = 100!!!!!!! wooooooooooooooop!   
          A_Tallying_BayesianTallying[p, i] <- sum(predictions.tallying[[i]]== dec_tallying)/length(test.labels) #should be 1 when penalty = 100       
          ## agreement with actual Linear Regression:
          A_LR_BayesianTTB[p, i]  <- sum(predictions.LR[[i]]==dec_TTB)/length(test.labels) # in the limit (penalty=100) LR_dec = TTB_dec?            
          A_LR_BayesianTallying[p, i]  <- sum(predictions.LR[[i]]==dec_tallying)/length(test.labels) # this one is good. is 1 when penalty = 0 :)               
          

          # test the limit cases
#           if (A_TTB_BayesianTTB[1,i] != 1 | A_Tallying_BayesianTallying[1, i] != 1){
#             stop('agreement is not 1 on this run!')
#           }
#           
  
      } #p loop
          
          
  } # k loop
        

        ## performance as a function of penalty parameter minus non-converged environments
          acc_dec_class <- apply(dec_class_acc, 1, mean)
          se_acc_class <- apply(dec_class_acc, 1, function(x) sd(x)/sqrt(k)) 
          acc_dec_TTB <- apply(dec_TTB_accs, 1, mean)
          se_acc_TTB <- apply(dec_TTB_accs, 1, function(x) sd(x)/sqrt(k)) 
          acc_dec_tallying <- apply(dec_tallying_accs, 1, mean)
          se_acc_tallying <- apply(dec_tallying_accs, 1, function(x) sd(x)/sqrt(k)) 
          
          ## performance as a function of penalty parameter minus non-converged environments
          acc.LR <- sum(results.linear.regression)/k
          se_acc_LR <- sd(results.linear.regression)/sqrt(k)
          acc.ttb <- sum(results.ttb)/k # always the same independent of p 
          se_ttb <- sd(results.ttb)/sqrt(k)
          acc.tallying <- sum(results.tallying)/k # always the same independent of p 
          se_tallying <- sd(results.tallying)/sqrt(k)
          
          Agreement_TTB_BayesianTTB <- apply(A_TTB_BayesianTTB, 1, mean) #rowSums(A_TTB_BayesianTTB)/k  
          se_TTB_BayesianTTB <- apply(A_TTB_BayesianTTB, 1, function(x) sd(x)/sqrt(k)) # sd(A_TTB_BayesianTTB[1, ])/sqrt(k)
          Agreement_Tallying_BayesianTallying <- apply(A_Tallying_BayesianTallying, 1, mean)  # is vector of length(penalty)
          se_Tallying_BayesianTallying <- apply(A_Tallying_BayesianTallying, 1, function(x) sd(x)/sqrt(k)) # sd(A_TTB_BayesianTTB[1, ])/sqrt(k)
          Agreement_LR_BayesianTTB <- apply(A_LR_BayesianTTB, 1, mean)  #
          se_LR_BayesianTTB <- apply(A_LR_BayesianTTB, 1, function(x) sd(x)/sqrt(k)) # sd(A_TTB_BayesianTTB[1, ])/sqrt(k)
          Agreement_LR_BayesianTallying <- apply(A_LR_BayesianTallying, 1, mean)  # is vector of length(penalty)
          se_LR_BayesianTallying <- apply(A_LR_BayesianTallying, 1, function(x) sd(x)/sqrt(k))
          

          cue_size = mean(cues)
          re = sum(re)/k
          ## --------------------- Write all results to a data frame and file ----------------------------------------------------    
          cv.run <- data.frame(guessing = 'no', data = paste(ALL_DATA[e]), v, percent_training, training_N = round(training_sample_size,0),  
                               modifications = 'zerovariance', 
                               penalty = penalty, Class_dec = acc_dec_class, se_acc_class = se_acc_class,   
                               acc_LR = acc.LR,  se_acc_LR = se_acc_LR, TTB_dec = acc_dec_TTB, se_acc_TTB = se_acc_TTB, 
                               Tallying_Dec = acc_dec_tallying, se_acc_tallying = se_acc_tallying,
                               TTB_Heuristic = acc.ttb, se_ttb_heuristic = se_ttb,
                               Tallying_Heuristic =  acc.tallying, se_ttb_heuristic = se_tallying,
                               Agreement_TTB_BayesianTTB = Agreement_TTB_BayesianTTB,  se_TTB_BayesianTTB =  se_TTB_BayesianTTB, 
                               Agreement_Tallying_BayesianTallying = Agreement_Tallying_BayesianTallying, 
                               se_Tallying_BayesianTallying = se_Tallying_BayesianTallying, 
                               Agreement_LR_BayesianTTB = Agreement_LR_BayesianTTB,
                               se_LR_BayesianTTB = se_LR_BayesianTTB, 
                               Agreement_LR_BayesianTallying = Agreement_LR_BayesianTallying, 
                               se_LR_BayesianTallying = se_LR_BayesianTallying, 
                               betas_LR = betas_LR ,#betas of whole dataset without penalization (matrix)
                               ind_coef  = ind_coeffs, # independent predictors at 0 for whole dataset (matrix)
                               Predictors, N, k = k,  avg_reshuffle = re, avg_cues = cue_size,  test_size = N - training_sample_size,
                               cov_mean = abs_mean_cor,cov_min = min_cor,cov_max = max_cor)    


          if (v > 1) cv.run <- rbind(results, cv.run) # append
          results  <- cv.run  
          
          # updates for every v loop advance
          save(results, file= paste("COR_", ALL_DATA[e],"_90%_ALLcues.RData", sep = ""))
          write.csv(results, file= paste("COR_", ALL_DATA[e],"_90%_ALLcues.csv", sep =""), row.names=TRUE)  
             

          } # v loop
          
      
        
    } # dopar e loop
