#Set Working Directory
setwd("C:/Users/utente/Documents/R/GitHub_Projects/Predict_Attrition_on_IBM_HR_Analytics_Dataset")

library(tictoc)
tic("Since Star_Time") #save start time

#Import packages
library(tidyverse)
library(readr)
library(dplyr)
library(data.table) 
library(readxl)
library(writexl)
library(ggplot2)

#Setup folders in the working directory
dir.create("Datasets")                      #save input datasets
dir.create("Datasets/Preprocessed_datasets")#save preprocessed datasets
dir.create("figs")                          #save plots' images
dir.create("figs/Correlation")              #Save Correlation plots


#Download and unzip file
fileURL <- "https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/download"

download.file(url = fileURL, 
              destfile = "../archive.zip")
unzip(zipfile = "../archive.zip" )

HR <- read_csv("./WA_Fn-UseC_-HR-Employee-Attrition.csv")
HR <- as.data.frame(HR)
str(HR)
dim(HR)

# Descriptive statistics
library(summarytools)
view(freq(HR))
view(descr(HR))
view(dfSummary(HR))

#Remove columns with one outcome
HR <- HR[,-c(9,22,27)]
names(HR)[9] <- "ID"

#Rename categorical outcomes
HR <- as.data.frame(HR %>%
                    mutate(BusinessTravel=case_when((BusinessTravel == "Non-Travel") ~ "No",
                                                    (BusinessTravel == "Travel_Frequently") ~ "Frequently",
                                                    (BusinessTravel== "Travel_Rarely") ~ "Rarely" )))
HR <- as.data.frame(HR %>%
                      mutate(Department=case_when((Department== "Human Resources") ~ "HR",
                                                  (Department == "Research & Development") ~ "R&D",
                                                  (Department == "Sales") ~ "Sales")))

HR <- as.data.frame(HR %>%
                      mutate(EducationField=case_when((EducationField == "Human Resources") ~ "HR",
                                                      (EducationField == "Life Sciences") ~ "LifeScience",
                                                      (EducationField == "Marketing") ~ "Marketing",
                                                      (EducationField == "Medical") ~ "Medical",
                                                      (EducationField == "Other") ~ "Other",
                                                      (EducationField == "Technical Degree") ~ "Technical")))                    
                    
HR <- as.data.frame(HR %>%
                      mutate(JobRole=case_when( (JobRole == "Healthcare Representative") ~ "HealthcareRep",
                                                (JobRole == "Human Resources") ~ "HR",
                                                (JobRole == "Laboratory Technician") ~ "LabTechnician",
                                                (JobRole == "Manager") ~ "Manager",
                                                (JobRole == "Manufacturing Director") ~ "ManufacturingDir",
                                                (JobRole == "Research Director") ~ "ResearchDir",
                                                (JobRole == "Research Scientist") ~ "ResearchSc",
                                                (JobRole == "Sales Executive") ~ "SalesExe",
                                                (JobRole == "Sales Representative") ~ "SalesRep")))

############################# Correlation ######################################
################################################################################
library(corrplot)
library(ggcorrplot)

HR_numeric <- select(HR[,-c(9)], where(is.numeric)) #Numeric variables
Correlation_matrix <- cor(HR_numeric) 
Correlation_matrix

Correlation_matrix_plot <- ggcorrplot(Correlation_matrix, 
                                      outline.col = "black",
                                      type = "upper",
                                      colors = c("#6D9EC1","white","#E46726"),
                                      lab = TRUE)
Correlation_matrix_plot


Correlation_matrix_plot <- corrplot(cor(HR_numeric),
                                    method="pie",
                                    type="upper",
                                    tl.cex = 0.75,
                                    tl.col = "black")
Correlation_matrix_plot


Correlation_matrix_plot <- corrplot(cor(HR_numeric),
                                    method="number",
                                    type="upper",
                                    tl.cex = 1,
                                    tl.col = "black",
                                    tl.srt = 45,
                                    cl.cex = 1,
                                    number.cex = 0.65)
Correlation_matrix_plot
ggsave("Correlation Matrix.png", path = "./figs/Correlation")


#Get highly correlated variable (Correlation >=0.55)
for (i in 1:nrow(Correlation_matrix)) {
  for (j in 1:ncol(Correlation_matrix)) {
    if(i<j && (Correlation_matrix[i,j]>=0.5|Correlation_matrix[i,j]<=-0.5)){
      print(paste(
        rownames(Correlation_matrix)[i],"is highly correlated with",colnames(Correlation_matrix)[j],
        "-> correlation is",round(Correlation_matrix[i,j],2)))
        }
  }
}

#Export preprocessed dataset as .xlsx file
write_xlsx(HR,".\\IBM HR Preprocessed.xlsx")

##final descriptive statistics
library(summarytools)
view(freq(HR))
view(descr(HR))
view(dfSummary(HR))

######################### Script Exectution time ###############################
################################################################################
toc()



