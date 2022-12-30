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

### 1) Dataset : About 150 pictures of 4 different facial expressions were collected on my iPhone. Thereafter, we wrote a function to crop only the face to reduce image noise. Below is an illustration.

![Unknown-2](https://user-images.githubusercontent.com/113895589/209963395-9ea233f1-483a-4451-b47f-014570d031bf.png)

### 2) Image Augmentation : Helps increase the size of the training dataset, helps the model generalize better to new, unseen data, helps the model learn more robust features and helps reduce overfitting

![Unknown-3](https://user-images.githubusercontent.com/113895589/209963762-9ef97394-dff4-4b1c-830c-20fc5c8ce44a.png)

### 3) Modeling

**Convolutional Neural Networks**

A convolutional neural network is a type of artificial neural network specifically designed for processing data that has a grid-like topology, such as an image. CNNs are particularly useful for image classification.

In a CNN, the input data is processed through multiple layers of interconnected nodes, each of which performs a specific operation on the data.

The layers of a CNN can include convolutional layers, pooling layers, and fully connected layers. Convolutional layers apply a set of filters to the input data, pooling layers reduce the dimensionality of the data, and fully connected layers learn a non-linear mapping between the input and output data.

Three different callbacks are used during the modeling process.
**1) Early Stopping**

Early stopping allows you to terminate the training process if the model's performance on a validation dataset stops improving. This can help you to avoid overfitting, where the model begins to memorize the training data rather than learning generalizable features.

**2) Model Checkpoint**

Model checkpointing is a technique used to save the state of a neural network during training. It allows you to save the model at regular intervals or at specific points during training, such as after completing a certain number of epochs or after achieving a certain level of accuracy.

**3) Learning Rate Scheduling**

Learning rate scheduling is a technique used to adjust the learning rate of a neural network during training. It can help the network converge faster and achieve better performance by adapting the learning rate to the specific characteristics of the training data.

- **Model 1 (Baseline)** : Simple CNN 

![Unknown-6](https://user-images.githubusercontent.com/113895589/210076845-6edfcf21-d269-4d0c-8898-57a0a26953f9.png)

The difference between the train and test accuracy is greater than 0.1 which shows that it is overfitting.

Additionally, based on the learning curves we can see that there is some overfitting where there is a big gap between the train and val accuracy. This trend was also shown on the loss function where there is a gap between train and val loss.

- **Model 2** : Model 1 + Dropout Layers

![Unknown-7](https://user-images.githubusercontent.com/113895589/210076863-d5eae05e-5ed7-4ea1-b083-c096f3129b2d.png)

Based on the learning curves, we can see that there is a smaller gap between the train and the val curves on both the accuracy and the loss function. This shows that our regularisation techniques has helped us to get a better fit.

By reducing overfitting, we can improve the generalizability of our model, as it is more likely to make accurate predictions on new, unseen data.

- **Model 3** : Model 2 + Hyperparameter Tuning

![Unknown-8](https://user-images.githubusercontent.com/113895589/210076871-5d31607b-88e2-42ba-b7ae-ef52a94958ff.png)

We can see that the train and test curves closely follows each other which means that it is a good fit. And after tuning the model, we managed to get a better model than before.

### 4) Final Model Score

<img width="550" alt="Screenshot 2022-12-30 at 8 41 06 PM" src="https://user-images.githubusercontent.com/113895589/210071196-80f3eb6c-e628-4a80-abc6-e20d850018df.png">

From the classification report, we can see that our model has a **recall score of 100% for angry faces**. 

### 5) Understanding the Model

**Correctly Predicted Images**

![Unknown-5](https://user-images.githubusercontent.com/113895589/210075570-930a8f7d-685e-4d64-97fd-ac621ff1b3dc.png)

**Wrongly Predicted Images**

![Unknown-4](https://user-images.githubusercontent.com/113895589/210075577-9cb5d207-2c6b-4af8-bc27-9eafe518a87b.png)

Based on the wrongly predicted images, most of the images are predicted as angry faces. I would infer that the eyebrows plays a big part in the model's prediction and since I have a more arched eyebrow by nature, there is a higher chance for it to be predicted as angry face when it is not.

**Model Activations**

<img width="835" alt="Screenshot 2022-12-30 at 9 44 26 PM" src="https://user-images.githubusercontent.com/113895589/210076681-dea3dbc2-abde-4a31-93e5-2bf704e5f956.png">

<img width="626" alt="Screenshot 2022-12-30 at 9 41 27 PM" src="https://user-images.githubusercontent.com/113895589/210076446-a1acbf77-6577-4d9a-bab6-59c18cb35d64.png">

When we take a closer look, eyes and eyebrows are the strongest features. By visualising the activations, we can see how an image is being processed through the model at different layers. We can see that after the second layer of convolution, we start to extract features such as the eyebrows and eyes as well as little bit of the lips. This further proves my inference that the model's prediction may be heavily influenced by the shape of the eyebrows.

### 6) Deploying the model on GoogleColab

Example shown in video above.

### Phase 1 Conclusion and Recommendations

**Conclusion**

To conclude this notebook, I would consider this to be a success. We have looked at many different ways and technique to help increase generalizability of our model. E.g. by cropping my face to reduce noise, applying image augmentation, using dropout layers as well as implementing earlystopping and model checkpoints.

We also managed to hit a recall for angry faces of 100% which surpasses our target of 80%. Overall, I would say that this model exceeded expectation.

**Limitations**

Some limitations I have is time, with more time, we will be able to test out even more complex models and tune even more parameters.

**Recommendations**

1) More photos and expressions could be implemented into the model to further increase generalizability of the model to unseen data.

2) Currently, this model is only trained using pictures of me. In the future, photos of others can be collected so that the model will be able to generalize on other types of faces and their facial expressions.

3) Use a different approach such as facial mapping.

## Phase 2 : Transfer Learning

#### Limitations

Time constraint: With more time, model can be trained on more epochs. However, since every epoch takes about 10minutes currently, we will be limiting our total epochs for this project to be 10.
