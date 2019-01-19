# Quora_Insincere_Questions_Classification
A text classification task using Deep Learning
## Analysis on data
0 - sincere,  1 - insincere

1. Imbalanced data set
![newplot](https://user-images.githubusercontent.com/40629085/51426221-64f8bf00-1c22-11e9-92df-35c7d0e99e27.png)

2. top 1-gram & 2-gram distributions
![newplot 1](https://user-images.githubusercontent.com/40629085/51426225-67f3af80-1c22-11e9-8471-5dff47e5a072.png)
![newplot 2](https://user-images.githubusercontent.com/40629085/51426222-65915580-1c22-11e9-863d-480a05890cc5.png)

## Text preprocessing
1. Convert all characters to lowercase characters
<img width="618" alt="2019-01-19 7 58 06" src="https://user-images.githubusercontent.com/40629085/51426365-8eb2e580-1c24-11e9-8bb9-fc70405e030f.png">

2. Standardize the format of punctuations(adding space before and after each punctuation)
<img width="729" alt="2019-01-19 7 55 31" src="https://user-images.githubusercontent.com/40629085/51426335-37ad1080-1c24-11e9-97e1-92264374ac2d.png">


## Blend of models approach
Before blending different models, the results of different models are as follows
<img width="397" alt="2019-01-19 7 47 02" src="https://user-images.githubusercontent.com/40629085/51426275-20b9ee80-1c23-11e9-8990-5563e8d12b68.png">

After blending the different results using simple linear regression, the resultant F1 score has greatly improved.
<img width="417" alt="2019-01-19 7 47 49" src="https://user-images.githubusercontent.com/40629085/51426273-1ef02b00-1c23-11e9-991f-2ce764b124eb.png">
