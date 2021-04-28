# Mutations
# Calculate bootstraped CI of accuracy, AUROC, AUPRC and other statistical metrics. Gather them into a single file.
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
library(gmodels)

# Check previously calculated trials
previous=read.csv("~/Documents/CCRCC_ITH/Results/Statistics_mutations.csv")
existed=paste(previous$Mutation, previous$Architecture, sep='_')
inlist = c("BAP1_p1", "BAP1_p2", "BAP1_p3", "BAP1_p4")
# Find the new trials to be calculated
targets = inlist[which(!inlist %in% existed)]
OUTPUT = setNames(data.frame(matrix(ncol = 50, nrow = 0)), c("Mutation", "Architecture", "Patient_ROC.95.CI_lower", "Patient_ROC",                 
                                                             "Patient_ROC.95.CI_upper",      "Patient_PRC.95.CI_lower",      "Patient_PRC",                  "Patient_PRC.95%CI_upper",      "Patient_Accuracy",            
                                                              "Patient_Kappa",                "Patient_AccuracyLower",        "Patient_AccuracyUpper",        "Patient_AccuracyNull",         "Patient_AccuracyPValue",      
                                                              "Patient_McnemarPValue",        "Patient_Sensitivity",          "Patient_Specificity",          "Patient_Pos.Pred.Value",       "Patient_Neg.Pred.Value",      
                                                              "Patient_Precision",            "Patient_Recall",               "Patient_F1",                   "Patient_Prevalence",           "Patient_Detection.Rate",      
                                                              "Patient_Detection.Prevalence", "Patient_Balanced.Accuracy",    "Tile_ROC.95.CI_lower",         "Tile_ROC",                     "Tile_ROC.95%CI_upper",        
                                                              "Tile_PRC.95.CI_lower",         "Tile_PRC",                     "Tile_PRC.95.CI_upper",         "Tile_Accuracy",                "Tile_Kappa",                  
                                                              "Tile_AccuracyLower",           "Tile_AccuracyUpper",           "Tile_AccuracyNull",            "Tile_AccuracyPValue",          "Tile_McnemarPValue",          
                                                              "Tile_Sensitivity",             "Tile_Specificity",             "Tile_Pos.Pred.Value",          "Tile_Neg.Pred.Value",          "Tile_Precision",              
                                                              "Tile_Recall",                  "Tile_F1",                      "Tile_Prevalence",              "Tile_Detection.Rate",          "Tile_Detection.Prevalence",   
                                                              "Tile_Balanced.Accuracy"))

# # PRC function for bootstrap
# auprc = function(data, indices){
#   sampleddf = data[indices,]
#   prc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
#   return(prc)
# }

for (i in targets){
  tryCatch(
    {
      print(i)
      folder = strsplit(i, '-')[[1]][1]  #split replicated trials
      folder_name = i  #get folder name
      pos = strsplit(i, '_')[[1]][1]  #get mutation
      arch = strsplit(i, '_')[[1]][2]  #get architecture used
      Test_slide <- read.csv(paste("~/documents/CCRCC_ITH/Results/", i, "/out/Test_slide.csv", sep=''))
      Test_tile <- read.csv(paste("~/documents/CCRCC_ITH/Results/", i, "/out/Test_tile.csv", sep=''))
      
      # per patient level
      answers <- factor(Test_slide$True_label)
      results <- factor(Test_slide$Prediction)
      # statistical metrics
      CMP = confusionMatrix(data=results, reference=answers, positive = pos)
      # ROC
      roc =  roc(answers, Test_slide$POS_score, levels=c('negative', pos))
      rocdf = t(data.frame(ci.auc(roc)))
      colnames(rocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
      # PRC
      SprcR = PRAUC(Test_slide$POS_score, factor(Test_slide$True_label))
      Sprls = list()
      for (j in 1:100){
        sampleddf = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.8)),]
        Sprc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
        Sprls[j] = Sprc
      }
      Sprcci = ci(as.numeric(Sprls))
      Sprcdf = data.frame('PRC.95.CI_lower' = Sprcci[2], 'PRC' = SprcR, 'PRC.95.CI_upper' = Sprcci[3])
      # Combine and add prefix
      soverall = cbind(rocdf, Sprcdf, data.frame(t(CMP$overall)), data.frame(t(CMP$byClass)))
      colnames(soverall) = paste('Patient', colnames(soverall), sep='_')
      
      # per tile level
      Tanswers <- factor(Test_tile$True_label)
      Tresults <- factor(Test_tile$Prediction)
      # statistical metrics
      CMT = confusionMatrix(data=Tresults, reference=Tanswers, positive = pos)
      # ROC
      Troc =  roc(Tanswers, Test_tile$POS_score, levels=c('negative', pos))
      Trocdf = t(data.frame(ci.auc(Troc)))
      colnames(Trocdf) = c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper')
      # PRC
      prcR = PRAUC(Test_tile$POS_score, factor(Test_tile$True_label))
      prls = list()
      for (j in 1:10){
        sampleddf = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.8)),]
        prc = PRAUC(sampleddf$POS_score, factor(sampleddf$True_label))
        prls[j] = prc
      }
      Tprcci = ci(as.numeric(prls))
      Tprcdf = data.frame('PRC.95.CI_lower' = Tprcci[2], 'PRC' = prcR, 'PRC.95.CI_upper' = Tprcci[3])
      # Combine and add prefix
      Toverall = cbind(Trocdf, Tprcdf, data.frame(t(CMT$overall)), data.frame(t(CMT$byClass)))
      colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
      # Key names
      keydf = data.frame("Mutation" = pos, "Architecture" = arch)
      # combine all df and reset row name
      tempdf = cbind(keydf, soverall, Toverall)
      rownames(tempdf) <- NULL
      OUTPUT = rbind(OUTPUT, tempdf)
    },
    error = function(error_message){
      message(error_message)
      message(i)
      return(NA)
    }
  )  
}

# Bind old with new; sort; save
New_OUTPUT = rbind(previous, OUTPUT)
New_OUTPUT = New_OUTPUT[order(New_OUTPUT$Mutation, New_OUTPUT$Architecture),]
write.csv(New_OUTPUT, file = "~/documents/CCRCC_ITH/Results/Statistics_mutations.csv", row.names=FALSE)

