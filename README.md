# Capstone Project : Human Emotion Recognition
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

1) Hotel Front Desk/Hotel Concierge. The human emotion recogniser can be deployed infront of hotel front desks or concierge to support hotel staff to identify angry customers. Once these angry customers are identified, the hotel can then send staff to investigate the issue and provide service recovery if necessary.

2) As hotel staff are humans too, the model can be deployed to face the hotel staff. Sometimes, when shift durations are long, hotel staff may start to lose the smile on their face. The model will then be able to detect and alert hotel staff to keep a smile on their face. Additionally, the model can also detect if a staff becomes angry at a customer. When that happens, the manager can be alerted to assist and resolve any conflicts the hotel staff may have with guests. 

3) Supermarket Aisles. The model can be deployed in supermarket aisles to identify whether or not a customer is facing anger such as not being able to find a certain product. The model can then detect and identify such angry customers and alert staff to assist with such customers.

4) As technology improves, more and more roles that used to require humans are replaced with robots. For example, airport check-in kiosks. This model can also be deployed on these robots to detect whenever there is any anger from the guests. Once identified, these establishments can then send staff to assist these angry guests.


## Contents

Part 1 : Phase 1 Modeling

Part 2 : Phase 1 Model Evaluation

Part 3 : Phase 1 Webcam Deployment on Googlecolab

Phase 2 : Transfer Learning with FER2013 (Kaggle Dataset)

## Phase 1 : Image Classification
https://user-images.githubusercontent.com/113895589/208602928-f1dc5be4-c20a-4523-a93d-a123f8e2688e.MOV

### 1) Dataset
About 150 pictures of 4 different facial expressions were collected on my iPhone. Thereafter, we wrote a function to crop only the face to reduce image noise. Below is an illustration.

![Unknown-2](https://user-images.githubusercontent.com/113895589/209963395-9ea233f1-483a-4451-b47f-014570d031bf.png)

### 2) Pre-processing + Image Augmentation

Pre-processing steps such as creation of train and test array, train-val split and normalisation are done.

Image Augmentation helps increase the size of the training dataset, helps the model generalize better to new, unseen data, helps the model learn more robust features and helps reduce overfitting

![Unknown-3](https://user-images.githubusercontent.com/113895589/209963762-9ef97394-dff4-4b1c-830c-20fc5c8ce44a.png)

### 3) Modeling

**Convolutional Neural Networks**

A convolutional neural network is a type of artificial neural network specifically designed for processing data that has a grid-like topology, such as an image. CNNs are particularly useful for image classification.

In a CNN, the input data is processed through multiple layers of interconnected nodes, each of which performs a specific operation on the data.

The layers of a CNN can include convolutional layers, pooling layers, and fully connected layers. Convolutional layers apply a set of filters to the input data, pooling layers reduce the dimensionality of the data, and fully connected layers learn a non-linear mapping between the input and output data.

**Three different callbacks are used during the modeling process.**

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

- **Model 3** : Model 2 + KerasTuner

![Unknown-8](https://user-images.githubusercontent.com/113895589/210076871-5d31607b-88e2-42ba-b7ae-ef52a94958ff.png)

<img width="328" alt="Screenshot 2022-12-30 at 10 01 29 PM" src="https://user-images.githubusercontent.com/113895589/210078361-45993655-c64d-4586-a50b-1e89d79262e5.png">

Using KerasTuner, we did a RandomSearch to find out how many units in the dense layers yields us the lowest val_loss. Thereafter, we rebuilt the model using the best parameters.

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

### 7) Phase 1 Conclusion and Recommendations

**Conclusion**

To conclude this notebook, I would consider this to be a success. We have looked at many different ways and technique to help increase generalizability of our model. E.g. by cropping my face to reduce noise, applying image augmentation, using dropout layers as well as implementing earlystopping and model checkpoints.

We also managed to hit a recall for angry faces of 100% which surpasses our target of 80%. Overall, I would say that this model exceeded expectation.

**Limitations**

Some limitations I have is time, with more time, we will be able to test out even more complex models and tune even more parameters.

**Recommendations**

1) More photos and expressions could be implemented into the model to further increase generalizability of the model to unseen data.

2) Currently, this model is only trained using pictures of me. In the future, photos of others can be collected so that the model will be able to generalize on other types of faces and their facial expressions.

3) Use a different approach such as facial mapping.

---

## Phase 2 : Transfer Learning with FER2013 Dataset using ResNet50

**What is transfer learning?**

Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task. Transfer learning is useful because it can allow a machine learning model to perform a new task more accurately and efficiently than if it were trained from scratch on that task. This is because the model can leverage the knowledge it has learned from the first task, and apply it to the second task, which can reduce the amount of data and compute resources required to train the model on the second task.

**Reason for using transfer learning?**

I decided to explore whether transfer learning could be useful as I wanted a model that was able to generalise to even more faces.

### 1) Dataset

The FER2013 contains about 33,000 images of 7 different facial expressions. As compared to our model in phase 1, the number of images in FER2013 is 55times more than the dataset in phase 1. 

![Unknown-9](https://user-images.githubusercontent.com/113895589/210130789-605b194d-d055-494e-af2d-3f1827a383a2.png)

### 2) Pre-processing + Image Augmentation

Below image visualises the image augmentation done to the images.

![Unknown-10](https://user-images.githubusercontent.com/113895589/210130806-ae2c03c7-54c3-4a1c-b045-997a7630075c.png)

### 3) Transfer Learning: Using a pre-trained model for feature extraction

In this step, we loaded a pre-trained model, ResNet50. Next, we built a model using feature extraction before fitting the model.

### 4) Fine Tuning

Fine-tuning a pre-trained model: To further improve performance, one might want to repurpose the top-level layers of the pre-trained models to the new dataset via fine-tuning. In this case, you tuned your weights such that your model learned high-level features specific to the dataset. This technique is usually recommended when the training dataset is large and very similar to the original dataset that the pre-trained model was trained on.

In this step, we unfreezed the top layers of the model, compiled the model and continued to train the model.

<img width="526" alt="Screenshot 2022-12-31 at 4 53 15 PM" src="https://user-images.githubusercontent.com/113895589/210131036-fda49ebf-291f-4949-9345-517ccf4b4017.png">

We can see that after fine-tuning, the model performance starts to improve. By unfreezing the top layers of a pre-trained model and training them on the new dataset, the model can learn task-specific features that are relevant to the second task, while still leveraging the knowledge it has learned from the first task. 

### 5) Model Evaluation

<img width="444" alt="Screenshot 2022-12-31 at 4 47 26 PM" src="https://user-images.githubusercontent.com/113895589/210130894-79c37dcb-3bd2-4e75-a2ba-d84bdfe3adcc.png">

Based on the classification report, we can see that the results are not as good as we expected. This is due to the fact that because the FER2013 dataset is such a huge dataset, it took a really long time to train. However, due to time constraints, we were not able to fully train the model to its ideal state. Additionally, callbacks and hyperparameter tuning can be done to further improve this model. And as mentioned, due to time constraints, this is the transfer learning model we currently have.

### 6) Conclusion and Recommendations

**Limitations**

Time constraint: With more time, model can be trained on more epochs. However, since the model takes a long time to train, we will be limiting our total epochs for this model to be 10.

Dataset: The FER2013 dataset is not representative of our use case. Most, if not all of the images are of people from different country from where this project is going to be deployed.

**Conclusion**

In conclusion, we have managed to explore the framework of building a model using transfer learning. However, due to the limitations, we will not be deploying this model. 

**Recommendations**

1) Try out different pre-trained models as in this part, we are only experimenting with ResNet50.

2) Collect our own dataset. However, such a huge dataset may require more time than the given deadline of this project.

3) Callbacks and Hyperparameter tuning can be done to further improve the model when time permits.

---
