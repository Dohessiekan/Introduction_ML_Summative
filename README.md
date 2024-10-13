# Credit Card Fraud Detection Project

Project Overview
This project aims to develop a robust machine learning model to detect fraudulent credit card transactions. The objective is to create an efficient and accurate fraud detection system that can identify potentially suspicious activities, thus helping financial institutions minimize losses and enhance transaction security.

Objectives
Build a reliable model for fraud detection using credit card transaction data.
Compare a simple vanilla neural network model with a more complex regularized neural network model.
Optimize model performance using regularization techniques and class weights.
Evaluate the model's effectiveness in identifying fraudulent transactions.
Dataset
Source: The dataset used for this project is the Credit Card Fraud Detection dataset from Kaggle.
Description: The dataset contains transactions made by credit cards, labeled as fraudulent or non-fraudulent. It includes several features that represent transaction details.
Preprocessing: Data was preprocessed to handle missing values, normalize features, and address class imbalance.
Key Findings
The Vanilla Neural Network(model1) served as a baseline model, showing moderate performance with standard neural network architecture.
The Regularized Neural Network (model2) significantly improved the accuracy and ability to generalize, utilizing regularization techniques and class weights to handle class imbalance.
Model2 demonstrated superior performance metrics, including accuracy, precision, and recall, making it more effective for detecting fraud.

#  Discussion on Optimization Techniques

In this Credit Card Fraud Detection project, we employed several optimization techniques to enhance the performance and accuracy of our models. The goal was to create a robust model capable of detecting fraudulent transactions with high precision. We implemented these techniques with a clear focus on regularization, class balancing, learning rate adaptation, and early stopping, each playing a critical role in improving the model's effectiveness.

1. Regularization Techniques
Regularization is a key method used to prevent overfitting by adding a penalty to the model's complexity. Overfitting occurs when a model learns the noise in the training data rather than the actual patterns, leading to poor generalization on unseen data.

L2 Regularization (Ridge Regression):

We applied L2 regularization in the hidden layers of our model (model2) with a penalty term (λ) of 0.01. L2 regularization works by adding the squared magnitude of the weights as a penalty to the loss function. This approach helps in keeping the weights small, thus preventing the model from becoming too complex.
Significance: It reduces the risk of overfitting by discouraging large weight values, leading to a simpler, more generalizable model.
Parameter Selection: The value of 0.01 was chosen after experimentation to strike a balance between regularization strength and model flexibility. A lower value would have had less impact on reducing complexity, while a higher value could have overly penalized the model, resulting in underfitting.
L1 Regularization (Lasso Regression):

L1 regularization was used in another hidden layer with a penalty term (λ) of 0.01. Unlike L2, L1 regularization adds the absolute magnitude of the weights as a penalty, encouraging sparsity in the model.
Significance: L1 regularization can lead to feature selection by driving some weights to zero, effectively removing less important features from the model.
Parameter Selection: Similar to L2, the λ value was set to 0.01 after tuning. This value ensured that the regularization was strong enough to promote sparsity without completely eliminating important features.
2. Dropout
Dropout is a technique used to prevent overfitting by randomly setting a fraction of the neurons to zero during the training process. In model2, we used dropout layers with a rate of 0.4.

Relevance: By temporarily ignoring a subset of neurons, dropout forces the network to learn more robust features that are less dependent on the presence of specific neurons.
Parameter Selection: The dropout rate of 0.4 was selected to provide a balance between retaining enough information for learning and preventing the network from becoming overly reliant on specific paths. Rates below 0.4 were found to be less effective in reducing overfitting, while higher rates led to under-utilization of the network's capacity.

3. Class Weights
Class imbalance is a common problem in fraud detection, where fraudulent transactions are far less frequent than non-fraudulent ones. To address this, we used class weights in model2, computed using Scikit-learn's compute_class_weight function.

Relevance: By assigning higher weights to the minority class (fraudulent transactions), the model pays more attention to correctly predicting these cases, improving recall and reducing false negatives.
Parameter Explanation: Class weights were dynamically calculated based on the distribution of the target classes. This approach ensures that the weights are tailored to the specific imbalance in the training data, enhancing the model's focus on detecting fraud.

4. Early Stopping
Early stopping is a technique used to halt training when the model's performance on a validation set stops improving, preventing overfitting.

Implementation: We set the early stopping criterion with a patience of 50 epochs, meaning that if the validation loss does not improve for 50 consecutive epochs, training is stopped.

Relevance: Early stopping ensures that the model training halts before it starts to overfit on the training data, saving computational resources and maintaining generalizability.
Parameter Selection: The patience value of 50 was chosen to allow the model enough time to explore improvements without prematurely stopping the training. This value was determined through multiple trials to optimize both model performance and training efficiency.

5. Learning Rate Optimization
Learning rate plays a critical role in determining the speed and convergence of the training process. For model2, we utilized learning rate scheduling along with the Adamax optimizer.
.Adamax Optimizer: An adaptive learning rate optimizer based on the Adam algorithm, particularly suited for large-scale sparse data. It combines the advantages of AdaGrad and RMSProp.
.Learning Rate Scheduling: We implemented a scheduler to reduce the learning rate when the validation loss plateaued, enhancing convergence.

.Relevance: These techniques help avoid overshooting the optimal point and ensure a smooth convergence towards the minimum loss.
.Parameter Justification: The initial learning rate and its decay schedule were tuned based on the validation performance, allowing for faster training while maintaining stability.



# ADITTIONNALITY

Instructions for Running the Notebook
Follow these steps to set up and run the project:

Set Up the Environment:

Ensure you have Python 3.7 or higher installed.
Install the required dependencies using:
bash
Copy code
pip install -r requirements.txt(import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from sklearn.metrics import classification_report)

# Download the Dataset:

Obtain the Credit Card Fraud Detection dataset from Kaggle.
Place the dataset in the appropriate directory within your project folder.
Running the Notebook:

Open the terminal and start Jupyter Notebook:
bash
Copy code
jupyter notebook
Navigate to and open the notebook.ipynb notebook.
Run each cell sequentially to execute data preprocessing, model training, and evaluation steps.
Loading the Saved Model:

Load the saved model (model2.pkl) without retraining:
python
Copy code
import joblib

# Load the saved model2
model2 = joblib.load('model2.pkl')
Model Evaluation:

Evaluate the loaded model using the test dataset and assess performance using metrics like accuracy, precision, and recall.
Making Predictions:

Predict fraud on new data with:
python
Copy code
predictions = model2.predict(new_data)
Conclusion
This project successfully demonstrates the impact of advanced model architectures and optimization techniques in enhancing fraud detection capabilities. By using both a vanilla model and a regularized model, we highlighted the importance of addressing class imbalance and overfitting for improved detection accuracy.
