# Random forest models reveal academic and financial factors outweigh demographics in predicting completion of a year-round veterinary program

### Authors:
Sarah E. Hooper, PhD<sup>1*</sup>; Natalie Ragland, DVM<sup>1–3</sup> ; Elpida Artemiou, PhD<sup>4,5</sup>  

<sup>1</sup> Department of Biomedical Sciences, School of Veterinary Medicine, Ross University, Basseterre, Saint Kitts and Nevis  
<sup>2</sup> Rowan-Virtua School of Translational Biomedical Engineering and Sciences, Rowan University, Stratford, NJ  
<sup>3</sup> Cooper Medical School, Rowan University, Camden, NJ  
<sup>4</sup> Department of Clinical Sciences, School of Veterinary Medicine, Ross University, Basseterre, Saint Kitts and Nevis  
<sup>5</sup> School of Veterinary Medicine, Texas Tech University, Amarillo, TX  
<sup>*</sup> Corresponding author: Dr. Hooper  

## Abstract
OBJECTIVE  
The purpose of this study was to develop random forest classifier models (a type of supervised machine learning algorithm)
that could (1) predict students who will or will not complete the DVM degree requirements and (2) identify
the top predictors for academic success and completion of the DVM degree.  
METHODS  
The study utilized Ross University School of Veterinary Medicine student records from 2013 to 2022. Twenty-four
variables encompassing demographic (eg, age, race), academic (eg, grade point average), and financial aid (eg,
outstanding balances) data were assessed in 11 cross-validated random forest machine learning models. One model
was built assessing all years of data and 10 individual models were developed for each enrollment year to compare
how the top predictors of success varied among the years.  
RESULTS  
Consistently, only academic and financial factors were identified as being features of importance (predictors) in all
models. Demographic factors such as race were not important for predicting student success. All models performed
very well to excellently based on multiple performance metrics including accuracy, ranging from 96.1% to 99%, and
the areas under the receiver operating characteristic curves, ranging from 98.1% to 99.9%.  
CONCLUSIONS  
The random forest algorithm is a powerful machine learning prediction model that performs well with veterinary
student academic records and is customizable such that variables important to each veterinary school’s student
population can be assessed.  
CLINICAL RELEVANCE  
Identifying predictors of success as well as at-risk students is essential for providing targeted curricular interventions
to increase retention and achieve timely completion of a DVM degree.  
Keywords: random forest, machine learning, veterinary medical education, underrepresented minority students,
social determinants  

### Overview

This repository contains example student data and the Python code to create one of the random forest machine learning models.  The code can be adapted to meet your analysis needs.  Please make sure to see the comments that help explain the code and point out important things such as ensuring your performance metrics use the testing datasets and not the entire dataset (training and testing).

[![DOI](https://zenodo.org/badge/582981344.svg)](https://doi.org/10.5281/zenodo.14051065)


