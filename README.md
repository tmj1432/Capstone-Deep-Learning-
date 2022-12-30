# Capstone Project : Human Emotion Recogniser
## Background
As someone who used to work in the service industry, it is easy for me to understand the importance of customer service and service recovery. Having such experience gave me an inspiration to apply my newly found Data Science knowledge to an industry where I used to come from.

## Problem Statement
In 2016, [Research](https://cloudblogs.microsoft.com/dynamics365/bdm/2016/07/22/4-frustrating-customer-service-experiences-and-how-to-fix-them/) shows that 60% of customers will stop doing business with a brand after just one poor customer service experience. However, [in a more recent study](https://www.forbes.com/sites/shephyken/2020/07/12/ninety-six-percent-of-customers-will-leave-you-for-bad-customer-service/?sh=64bd471d30f8), it shows that 96% of customers will leave you for bad customer service. Additionally, when it comes to reviews, the fact is that people are more likely to share their bad reviews than their good ones. However, most of these lost customers could have been retained had the problem been resolved or by providing service recovery before it is too late. This should help you to understand why it is so important for a business to provide good customer service.

Today, I will be building a deep learning model that will help identify angry customers so that businesses will be able to take action to appease and provide service recovery to these angry customers before they leave the establishment.

## Success Metric

As we are building a model for the use in the customer service industry, it would mean that being able to correctly identify angry customers is important. With the ability to identify such customers, establishments are able to provide service recovery before it is too late

As recall is the fraction of positive instances that are correctly predicted by the classifier, we want as many correctly predicted angry faces as possible. This is due to the fact that there are more consequences of not being able to identify angry faces.

Therefore, we are looking to achieve greater than 80% recall for Angry Faces.

## Example use cases

1) 

## Contents

Part 1 : Phase 1 Modeling

Part 2 : Phase 1 Model Evaluation

Part 3 : Phase 1 Webcam Deployment on Googlecolab

Phase 2 : Transfer Learning with FER2013 (Kaggle Dataset)

## Phase 1 : Image Classification
https://user-images.githubusercontent.com/113895589/208602928-f1dc5be4-c20a-4523-a93d-a123f8e2688e.MOV

1) Dataset : About 150 pictures of 4 different facial expressions were collected on my iPhone. Thereafter, we wrote a function to crop only the face to reduce image noise. Below is an illustration.

![Unknown-2](https://user-images.githubusercontent.com/113895589/209963395-9ea233f1-483a-4451-b47f-014570d031bf.png)

2) Image Augmentation : Helps increase the size of the training dataset, helps the model generalize better to new, unseen data, helps the model learn more robust features and helps reduce overfitting

![Unknown-3](https://user-images.githubusercontent.com/113895589/209963762-9ef97394-dff4-4b1c-830c-20fc5c8ce44a.png)

3) Modeling

- Model 1 (Baseline) : Simple CNN 
- Model 2 : Model 1 + Dropout Layers
- Model 3 : Model 2 + Hyperparameter Tuning

4) Final Model Score

<img width="550" alt="Screenshot 2022-12-30 at 8 41 06 PM" src="https://user-images.githubusercontent.com/113895589/210071196-80f3eb6c-e628-4a80-abc6-e20d850018df.png">

From the classification report, we can see that our model has a **recall score of 100% for angry faces**. 

5) Understanding the Model

6) Deploying the model on GoogleColab

Example shown in video above.

## Phase 2 : Transfer Learning

#### Limitations

Time constraint: With more time, model can be trained on more epochs. However, since every epoch takes about 10minutes currently, we will be limiting our total epochs for this project to be 10.
