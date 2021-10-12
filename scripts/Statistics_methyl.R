# Subtypes
# Calculate bootstraped CI of accuracy, AUROC, AUPRC and other statistical metrics. Gather them into a single file.
library(readxl)
library(caret)
library(pROC)
library(dplyr)
library(MLmetrics)
library(boot)
library(gmodels)


previous=read.csv("~/Documents/CCRCC_ITH/Results/Statistics_methyl.csv")
existed=paste("met", previous$Architecture, sep='_')
inlist = c('met_p1', 'met_p2', 'met_p3', 'met_p4')
# Find the new trials to be calculated
targets = inlist[which(!inlist %in% existed)]
OUTPUT = setNames(data.frame(matrix(ncol = 123, nrow = 0)), c("Architecture"                                                                , "Patient_Multiclass_ROC.95.CI_lower"   ,    
                                                                "Patient_Multiclass_ROC"                   , "Patient_Multiclass_ROC.95.CI_upper"       , "Patient_me1_ROC.95.CI_lower"          ,    
                                                                "Patient_me1_ROC"                          , "Patient_me1_ROC.95.CI_upper"              , "Patient_me2_ROC.95.CI_lower" ,    
                                                                "Patient_me2_ROC"                 , "Patient_me2_ROC.95.CI_upper"     , "Patient_me3_ROC.95.CI_lower"  ,    
                                                                "Patient_me3_ROC"                  , "Patient_me3_ROC.95.CI_upper"     , "Patient_me1_PRC.95.CI_lower"          ,    
                                                                "Patient_me1_PRC"                          , "Patient_me1_PRC.95.CI_upper"              , "Patient_me2_PRC.95.CI_lower" ,    
                                                                "Patient_me2_PRC"                 , "Patient_me2_PRC.95.CI_upper"     , "Patient_me3_PRC.95.CI_lower"  ,    
                                                                "Patient_me3_PRC"                  , "Patient_me3_PRC.95.CI_upper"      , "Patient_Accuracy"                     ,    
                                                                "Patient_Kappa"                            , "Patient_AccuracyLower"                    , "Patient_AccuracyUpper"                ,    
                                                                "Patient_AccuracyNull"                     , "Patient_AccuracyPValue"                   , "Patient_McnemarPValue"                ,    
                                                                "Patient_me2_Sensitivity"         , "Patient_me2_Specificity"         , "Patient_me2_Pos.Pred.Value"  ,    
                                                                "Patient_me2_Neg.Pred.Value"      , "Patient_me2_Precision"           , "Patient_me2_Recall"          ,    
                                                                "Patient_me2_F1"                  , "Patient_me2_Prevalence"          , "Patient_me2_Detection.Rate"  ,    
                                                                "Patient_me2_Detection.Prevalence", "Patient_me2_Balanced.Accuracy"   , "Patient_me1_Sensitivity"              ,    
                                                                "Patient_me1_Specificity"                  , "Patient_me1_Pos.Pred.Value"               , "Patient_me1_Neg.Pred.Value"           ,    
                                                                "Patient_me1_Precision"                    , "Patient_me1_Recall"                       , "Patient_me1_F1"                       ,    
                                                                "Patient_me1_Prevalence"                   , "Patient_me1_Detection.Rate"               , "Patient_me1_Detection.Prevalence"     ,    
                                                                "Patient_me1_Balanced.Accuracy"            ,  
                                                                "Patient_me3_Sensitivity"          , "Patient_me3_Specificity"          , "Patient_me3_Pos.Pred.Value"   ,    
                                                                "Patient_me3_Neg.Pred.Value"       , "Patient_me3_Precision"            , "Patient_me3_Recall"           ,    
                                                                "Patient_me3_F1"                   , "Patient_me3_Prevalence"           , "Patient_me3_Detection.Rate"   ,    
                                                                "Patient_me3_Detection.Prevalence" , "Patient_me3_Balanced.Accuracy"    , "Tile_Multiclass_ROC.95.CI_lower"      ,    
                                                                "Tile_Multiclass_ROC"                      , "Tile_Multiclass_ROC.95.CI_upper"          , "Tile_me1_ROC.95.CI_lower"             ,    
                                                                "Tile_me1_ROC"                             , "Tile_me1_ROC.95.CI_upper"                 , "Tile_me2_ROC.95.CI_lower"    ,    
                                                                "Tile_me2_ROC"                    , "Tile_me2_ROC.95.CI_upper"        , "Tile_me3_ROC.95.CI_lower"     ,    
                                                                "Tile_me3_ROC"                     , "Tile_me3_ROC.95.CI_upper"         , "Tile_me1_PRC.95.CI_lower"             ,    
                                                                "Tile_me1_PRC"                             , "Tile_me1_PRC.95.CI_upper"                 , "Tile_me2_PRC.95.CI_lower"    ,    
                                                                "Tile_me2_PRC"                   ,  "Tile_me2_PRC.95.CI_upper"       ,  "Tile_me3_PRC.95.CI_lower"    ,     
                                                                "Tile_me3_PRC"                    ,  "Tile_me3_PRC.95.CI_upper"        ,  "Tile_Accuracy"                       ,     
                                                                "Tile_Kappa"                              ,  "Tile_AccuracyLower"                      ,  "Tile_AccuracyUpper"                  ,     
                                                                "Tile_AccuracyNull"                       ,  "Tile_AccuracyPValue"                     ,  "Tile_McnemarPValue"                  ,     
                                                                "Tile_me2_Sensitivity"           ,  "Tile_me2_Specificity"           ,  "Tile_me2_Pos.Pred.Value"    ,     
                                                                "Tile_me2_Neg.Pred.Value"        ,  "Tile_me2_Precision"             ,  "Tile_me2_Recall"            ,     
                                                                "Tile_me2_F1"                    ,  "Tile_me2_Prevalence"            ,  "Tile_me2_Detection.Rate"    ,     
                                                                "Tile_me2_Detection.Prevalence"  ,  "Tile_me2_Balanced.Accuracy"     ,  "Tile_me1_Sensitivity"                ,     
                                                                "Tile_me1_Specificity"                    ,  "Tile_me1_Pos.Pred.Value"                 ,  "Tile_me1_Neg.Pred.Value"             ,     
                                                                "Tile_me1_Precision"                      ,  "Tile_me1_Recall"                         ,  "Tile_me1_F1"                         ,     
                                                                "Tile_me1_Prevalence"                     ,  "Tile_me1_Detection.Rate"                 ,  "Tile_me1_Detection.Prevalence"       ,     
                                                                "Tile_me1_Balanced.Accuracy"             ,     
                                                                "Tile_me3_Sensitivity"            ,  "Tile_me3_Specificity"            ,  "Tile_me3_Pos.Pred.Value"     ,     
                                                                "Tile_me3_Neg.Pred.Value"         ,  "Tile_me3_Precision"              ,  "Tile_me3_Recall"             ,     
                                                                "Tile_me3_F1"                     ,  "Tile_me3_Prevalence"             ,  "Tile_me3_Detection.Rate"     ,     
                                                                "Tile_me3_Detection.Prevalence"   ,  "Tile_me3_Balanced.Accuracy"))



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
      arch = strsplit(i, '_')[[1]][2]  #get architecture used
      Test_slide <- read.csv(paste("~/documents/CCRCC_ITH/Results/", i, "/out/Test_slide.csv", sep=''))
      Test_tile <- read.csv(paste("~/documents/CCRCC_ITH/Results/", i, "/out/Test_tile.csv", sep=''))
      
      # per patient level
      answers <- factor(Test_slide$True_label)
      results <- factor(Test_slide$Prediction)
      # statistical metrics
      CMP = confusionMatrix(data=results, reference=answers)
      dddf = data.frame(t(CMP$overall))
      for (m in 1:3){
        temmp = data.frame(t(CMP$byClass[m,]))
        colnames(temmp) = paste(gsub('-', '\\.', strsplit(rownames(CMP$byClass)[m], ': ')[[1]][2]), colnames(temmp), sep='_')
        dddf = cbind(dddf, temmp)
      }
      
      # multiclass ROC
      score = select(Test_slide, me1_score, me2_score, me3_score)
      colnames(score) = c("me1", "me2", "me3")
      roc =  multiclass.roc(answers, score)$auc
      rocls=list()
      for (q in 1:100){
        sampleddf = Test_slide[sample(nrow(Test_slide), round(nrow(Test_slide)*0.8)),]
        score = select(sampleddf, me1_score, me2_score, me3_score)
        colnames(score) = c("me1", "me2", "me3") 
        answers <- factor(sampleddf$True_label)
        rocls[q] = multiclass.roc(answers, score)$auc
      }
      rocci = ci(as.numeric(rocls))
      mcroc = data.frame('Multiclass_ROC.95.CI_lower' = rocci[2], 'Multiclass_ROC' = roc, 'Multiclass_ROC.95.CI_upper' = rocci[3])
      
      rocccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      prcccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      #ROC and PRC
      for (w in 2:4){
        cpTest_slide = Test_slide
        case=strsplit(colnames(cpTest_slide)[w], '_')[[1]][1]
        case = gsub('\\.', '-', c(case))
        cpTest_slide$True_label = as.character(cpTest_slide$True_label)
        cpTest_slide$True_label[cpTest_slide$True_label != case] = "negative"
        cpTest_slide$True_label = as.factor(cpTest_slide$True_label)
        answers <- factor(cpTest_slide$True_label)
        
        #ROC
        roc =  roc(answers, cpTest_slide[,w], levels = c("negative", case))
        rocdf = t(data.frame(ci.auc(roc)))
        colnames(rocdf) = paste(gsub('-', '\\.', c(case)), c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper'), sep='_')
        rocccc = cbind(rocccc, rocdf)
        
        # PRC
        SprcR = PRAUC(cpTest_slide[,w], answers)
        Sprls = list()
        for (j in 1:100){
          sampleddf = cpTest_slide[sample(nrow(cpTest_slide), round(nrow(cpTest_slide)*0.95)),]
          Sprc = PRAUC(sampleddf[,w], factor(sampleddf$True_label))
          Sprls[j] = Sprc
        }
        Sprcci = ci(as.numeric(Sprls))
        Sprcdf = data.frame('PRC.95.CI_lower' = Sprcci[2], 'PRC' = SprcR, 'PRC.95.CI_upper' = Sprcci[3])
        colnames(Sprcdf) = paste(gsub('-', '\\.', c(case)), colnames(Sprcdf), sep='_')
        prcccc = cbind(prcccc, Sprcdf)
      }
      
      # Combine and add prefix
      soverall = cbind(mcroc, rocccc, prcccc, dddf)
      colnames(soverall) = paste('Patient', colnames(soverall), sep='_')
      
      
      
      # per tile level
      answers <- factor(Test_tile$True_label)
      results <- factor(Test_tile$Prediction)
      # statistical metrics
      CMT = confusionMatrix(data=results, reference=answers)
      Tdddf = data.frame(t(CMT$overall))
      for (m in 1:3){
        Ttemmp = data.frame(t(CMT$byClass[m,]))
        colnames(Ttemmp) = paste(gsub('-', '\\.', strsplit(rownames(CMT$byClass)[m], ': ')[[1]][2]), colnames(Ttemmp), sep='_')
        Tdddf = cbind(Tdddf, Ttemmp)
      }
      
      # multiclass ROC
      Tscore = select(Test_tile, me1_score, me2_score, me3_score)
      colnames(Tscore) = c("me1", "me2", "me3")
      Troc =  multiclass.roc(answers, Tscore)$auc
      Trocls=list()
      for (q in 1:10){
        Tsampleddf = Test_tile[sample(nrow(Test_tile), round(nrow(Test_tile)*0.8)),]
        Tscore = select(Tsampleddf, me1_score, me2_score, me3_score)
        colnames(Tscore) = c("me1", "me2", "me3") 
        Tanswers <- factor(Tsampleddf$True_label)
        Trocls[q] = multiclass.roc(Tanswers, Tscore)$auc
      }
      Trocci = ci(as.numeric(Trocls))
      Tmcroc = data.frame('Multiclass_ROC.95.CI_lower' = Trocci[2], 'Multiclass_ROC' = Troc, 'Multiclass_ROC.95.CI_upper' = Trocci[3])
      
      Trocccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      Tprcccc = as.data.frame(matrix(0, ncol = 0, nrow = 1))
      #ROC and PRC
      for (w in 7:9){
        cpTest_tile = Test_tile
        case=strsplit(colnames(cpTest_tile)[w], '_')[[1]][1]
        case = gsub('\\.', '-', c(case))
        cpTest_tile$True_label = as.character(cpTest_tile$True_label)
        cpTest_tile$True_label[cpTest_tile$True_label != case] = "negative"
        cpTest_tile$True_label = as.factor(cpTest_tile$True_label)
        Tanswers <- factor(cpTest_tile$True_label)
        
        #ROC
        Troc =  roc(Tanswers, cpTest_tile[,w], levels = c("negative", case))
        Trocdf = t(data.frame(ci.auc(Troc)))
        colnames(Trocdf) = paste(gsub('-', '\\.', c(case)), c('ROC.95.CI_lower', 'ROC', 'ROC.95.CI_upper'), sep='_')
        Trocccc = cbind(Trocccc, Trocdf)
        
        # PRC
        TprcR = PRAUC(cpTest_tile[,w], Tanswers)
        Tprls = list()
        for (j in 1:10){
          sampleddf = cpTest_tile[sample(nrow(cpTest_tile), round(nrow(cpTest_tile)*0.95)),]
          Tprc = PRAUC(sampleddf[,w], factor(sampleddf$True_label))
          Tprls[j] = Tprc
        }
        Tprcci = ci(as.numeric(Tprls))
        Tprcdf = data.frame('PRC.95.CI_lower' = Tprcci[2], 'PRC' = TprcR, 'PRC.95.CI_upper' = Tprcci[3])
        colnames(Tprcdf) = paste(gsub('-', '\\.', c(case)), colnames(Tprcdf), sep='_')
        Tprcccc = cbind(Tprcccc, Tprcdf)
      }
      
      # Combine and add prefix
      Toverall = cbind(Tmcroc, Trocccc, Tprcccc, Tdddf)
      colnames(Toverall) = paste('Tile', colnames(Toverall), sep='_')
      
      # Key names
      keydf = data.frame("Architecture" = arch)
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
New_OUTPUT = New_OUTPUT[order(New_OUTPUT$Architecture),]
write.csv(New_OUTPUT, file = "~/documents/CCRCC_ITH/Results/Statistics_methyl.csv", row.names=FALSE)
