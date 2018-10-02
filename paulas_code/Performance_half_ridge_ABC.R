#
## preamble 

    rm(list=ls())
    
    packages <- c('clusterGeneration', 'MASS', 'ggplot2', 'lars', 'psych', 'matrixcalc', 'abind', 'lattice', 'coda', 
                  'foreach', 'doParallel', 'tmvtnorm')
    lapply(packages, require, character.only = TRUE)
    
      
    source("Functions_halfridge_abc.R")
    
    #Set to the number of cores (Brad has 12)
    #registerDoMC(2)
    registerDoParallel(cores= 12)   


    ALL_DATA <- c("house.world","mortality","cit.world","prf.world","bodyfat.world", "car.world","cloud",
                  "dropout","fat.world", "fuel.world", "glps",
                  "homeless.world", "landrent.world", "mammal.world", "oxidants",
                  "attractiveness.men", "attractiveness.women", "fish.fertility","oxygen", "ozone")
         
    #all_results<- vector("list", 12)

    #Run a parallel loop, each iteration should be a different job you need doing (e.g. fitting your model with different starting parameters)
#     all_results<-foreach(e=1:12) %dopar%   
#     { 



      for (e in 1:20){
  
      ##-------------------------- Imread datasets  ---------------------------------------------------------
      dataset <- read.table(paste(ALL_DATA[e],".txt", sep = ""), header = TRUE) # no space allowed
      
      
      # quicker than excel: another 1e07 for convergence
      penalty <- c(10000000, 1000000, 1000, 700.00000000, 330.07641174, 156.81303712,  74.49889703,  35.39301171,  16.81454797,   7.98827254,
                   3.79507663,   1.80296886, 0.85655628,   0.40693363,   0.19332644,   0.09184572)
       
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
      k <- 300 # test run with 5 to see if it works # larger but also less training set size  (5)
      
      # -----------------------------------------------------------------------------------------------------
      ## BEFORE creating paired_data: average redundancy between cues
      cov <- as.matrix(dataset[,col_cues])
      cor_mat <- cor(cov,method = "pearson")
      # the lower triangle of the cov matrix contains pairwise correlations
      min_cor <- min(cor_mat[lower.tri(cor_mat)])
      max_cor <- max(cor_mat[lower.tri(cor_mat)])
      abs_mean_cor <- mean(abs(cor_mat[lower.tri(cor_mat)]))
      # -------------------------------------------------------------------------------------------------
      
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
      
      
## WITH OR WITHOUT: evening out the 2 class frequencies: 
      ### have prior of 0.5 for either class, its like taking into account an intercept, should make performance  of linear model better, and training sizes perform normal
#       classA <- sum(paired_data$dependent == 1) # 100
#       classA_data <- paired_data[paired_data$dependent == 1,]    
#       classB <- sum(paired_data$dependent == -1) # 210 
#       classB_data <- paired_data[paired_data$dependent == -1,] 
#       
#       if (classA > classB) {
#         n <- classA -classB
#         perm <- sample(1:classA, classA) # as long as classA (1) data part
#         classA_data <- classA_data[-perm[1:n], ] 
#         paired_data <- rbind(classA_data, classB_data)
#         
#       } else if (classB > classA){
#         n <- classB - classA  # difference
#         perm <- sample(1:classB, classB) # as long as classB (-1) data part
#         classB_data <- classB_data[-perm[1:n], ] # sample exactly n rows from the overall paired_data and delete
#         paired_data <- rbind(classA_data, classB_data)
#       }
#       
#       # number of objects
#       N <- nrow(paired_data)
#       
#       
      
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
      
      #--------------------------------------------------------------------------------------------------------
      dataset <- paired_data # is the new dataset now for below
      
      
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
      
      
# 10 is as good or better than 5 
      training_size <- c(10, 20, 115)

      
      if(ALL_DATA[e] == "oxygen"){
        training_size[3] <- 72 # 20 test set
      } else if (ALL_DATA[e] == "ozone"){
        training_size[3] <- 35 # 20 test set
      } 
      # precentage
      training_size <- training_size/N
      

      chain_length <- vector('numeric', length(training_size))
      chain_length[1] <- 20000
      chain_length[2] <- 20000
      chain_length[3] <- 100000

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
        halfridge_acc <- matrix(nrow=length(penalty), ncol = k)
        A_Tallying_halfridge <- matrix(nrow=length(penalty), ncol = k)
        A_LR_halfridge <- matrix(nrow=length(penalty), ncol = k)
         
        predictions.LR <- vector('list',k) 
        predictions.ttb <- vector('list',k) 
        predictions.tallying <- vector('list',k) 
        predictions.Trunctallying  <- vector('list',k) 
        
        results.linear.regression <- vector('numeric', k)
        results.ttb <- vector('numeric', k)
        results.tallying <- vector('numeric', k)
        results.trunc.tallying <- vector('numeric', k)
        re <- vector('numeric', k) # resampling
        cues <- vector('numeric', k) 
        
        for (i in 1:k){  # test
          
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
          cat(paste("\n\nIteration  = ", i))     
          cues[i] <- Predictors     
          
          test <- as.matrix(testset[ ,-ncol(testset)]) 
          test.labels <- testset[ ,ncol(testset)]
          # Predictors needs to be updated in each loop
          x <- as.matrix(trainset[ ,1:Predictors])    
          y <- trainset$dependent
          
          
          ###--- ------------------  Regular Linear Regression: MAKE SURE IT USES manual (ginv) computation as well!
          # using ginv inverse function 
          predictions.LR[[i]] <- Linear.regression(x, y, test, Predictors, labels) #     
          #predictions.trunc_LR[[i]] <- truncated.linear(Predictors,  x, y, penalty = 0, test, sigma_2 = 0.25) #     
          
          ###--- ------------------  Regular TTB Heuristic ------------------------------------##################################### 
          predictions.ttb[[i]] <- ttb.predictions(trainset, test, Predictors)
          ###--- ------------------  Regular Tallying Heuristic ------------------------------------##################################### 
          predictions.tallying[[i]] <- tallying.learning(trainset, test, Predictors)      
          ###--- ------------------  Truncated Tallying Heuristic 
          predictions.Trunctallying[[i]] <- truncated.tallying(trainset, test, Predictors)
          

          # get TTB Heuristic accuracy
          results.ttb[i] <- ttb.graph(predictions.ttb[[i]], test.labels)
          # get TTB Heuristic accuracy
          results.tallying[i] <- ttb.graph(predictions.tallying[[i]], test.labels)
          # get LR accuracy
          results.linear.regression[i] <- regression.graph(predictions.LR[[i]], test.labels)
          # get Trunc Tall accuracy
          results.trunc.tallying[i] <- regression.graph(predictions.Trunctallying[[i]], test.labels)
          
           
          ###---------------------------- Half-Ridge (L2) Model --------------------------------------------------------------------------------

          meanW <- vector('list',length(penalty)) 
          for (p in 1: length(penalty)){
            
            sigma_2 <- 0.25 # try different values
            
            cat(paste("\n\npenalty  = ", p))   
            
            meanW[[p]] <- halfridge(Predictors,x, y,penalty[p], sigma_2, chain_length[v])
            # these weights are already the re-scaled ones so that it converges to tallying in limit
            
            ## Rounding precision in limit matters as ratio of w/w has to be 1
             w <- round(meanW[[p]],4)
                      
          
            # Matrix multiplication with testset, and take the sign to see whether A or B is predicted
            outcomes <-   test %*%  w
         
            
            ## For limit convergence: cant do this earlier because for weights are near zero for small penalties 
            if (p == 1 | p == 2){
              outcomes <- round(outcomes,0)
            }  # by doing this we get sudden agreement of 1 for end penalties but not smooth cureves. should do one rule for all penalty parameters
            
            
            outcomes <- sign(outcomes)
            
            ## For Tallying/halfridge: # roundign ensures that if two weights are 0.81000 and 0.79963 they give 0 still
            #outcomes <- sign(round(outcomes, 1))
            
            # making predictions to be either y = -1 or 1
#             for (j in 1:length(outcomes)){  
#               if (outcomes[j]==0)
#               {
#                 outcomes[j] <- sign(rnorm(1))
#               }
#             }   ## all +1 and -1 now!
#             
#             
      
            # convergence agreement 
            A_Tallying_halfridge[p, i] <- sum(predictions.Trunctallying[[i]]== outcomes)/length(test.labels) 

          
            # performance should be equal to tallying truncated!
            halfridge_acc[p,i] <- sum(outcomes == test.labels)/length(test.labels)
           
            #test the limit cases
                      if (A_Tallying_halfridge[1,i] != 1){
                        stop('agreement is not 1 on this run!')
                      }
                      
            
          } #p loop
          
          
        } # k loop
  

  
      # Get rid of NA runs: which columns of k columns can be used and index them
        valid <- which(!is.na(colSums(halfridge_acc))) # is the same columns for A_Tallying_halfridge
        S <- length(valid)

        acc_halfridge <-  apply(halfridge_acc[  ,valid], 1, mean) 
        se_halfridge <- apply(halfridge_acc[  ,valid], 1, function(x) sd(x)/sqrt(S)) 
        
        ## performance as a function of penalty parameter minus non-converged environments
        acc.LR <- sum(results.linear.regression[valid])/S
        se_acc_LR <- sd(results.linear.regression[valid])/sqrt(S)
        acc.ttb <- sum(results.ttb[valid])/S # always the same independent of p 
        se_ttb <- sd(results.ttb[valid])/sqrt(S)
        acc.tallying <- sum(results.tallying[valid])/S # always the same independent of p 
        se_tallying <- sd(results.tallying[valid])/sqrt(S)
        acc.trunc.tallying <- sum(results.trunc.tallying[valid])/S # always the same independent of p 
        se_trunc.tallying <- sd(results.trunc.tallying[valid])/sqrt(S)

        # Agreement
        Agreement_Tallying_Halfridge <- apply(A_Tallying_halfridge[  ,valid], 1, mean) #rowSums(A_TTB_BayesianTTB)/S  
        se_Tallying_Halfridge <- apply(A_Tallying_halfridge[  ,valid], 1, function(x) sd(x)/sqrt(S)) # sd(A_TTB_BayesianTTB[1, ])/sqrt(S)
       

        cue_size = mean(cues[valid])
        re = mean(re[valid])
        ## --------------------- Write all results to a data frame and file ----------------------------------------------------    
        cv.run <- data.frame(guessing = 'no', data = paste(ALL_DATA[e]), v, percent_training, training_N = round(training_sample_size,0),  
                             converged_samples = S, 
                             penalty = penalty, acc_halfridge = acc_halfridge, se_halfridge = se_halfridge, 
                             acc.trunc.tallying =  acc.trunc.tallying, se_trunc.tallying = se_trunc.tallying,                       
                             Tallying_Heuristic =  acc.tallying,se_tallying_heuristic = se_tallying,    
                             
                             acc_LR = acc.LR,  se_acc_LR = se_acc_LR, 
                             TTB_Heuristic = acc.ttb, se_ttb_heuristic = se_ttb,
                             Agreement_Tallying_Halfridge = Agreement_Tallying_Halfridge,  
                             se_Tallying_Halfridge =  se_Tallying_Halfridge, 
                           
                             
                             betas_LR = betas_LR ,#betas of whole dataset without penalization (matrix)
                             ind_coef  = ind_coeffs, # independent predictors at 0 for whole dataset (matrix)
                             Predictors, N, k = k,  avg_reshuffle = re, avg_cues = cue_size,  test_size = N - training_sample_size,
                             cov_mean = abs_mean_cor,cov_min = min_cor,cov_max = max_cor)    
        
      
      if (v > 1) {cv.run <- rbind(results, cv.run)} # append
      results  <- cv.run  
      
      # updates for every v loop advance
      save(results, file= paste("HalfRidge_", ALL_DATA[e],"_gibbs_NaNcorrected.RData", sep = ""))
      write.csv(results, file= paste("HalfRidge_", ALL_DATA[e],"_gibbs_NaNcorrected.csv", sep =""), row.names=TRUE)  

        
      } # v loop
      
      # Return object that is output of the dopar function
      # summary for one data set: all v conditions
      #results 
      
      
    
    }  # dopar (e) loop
