# ML-accelerate-deep-eutectic-solvent-discovery
code for article "Machine learning models accelerate deep eutectic solvent discovery for the recycling of lithium-ion battery cathodes"

Overview:
The goal of this project is to accelerate the discovery of novel promising Deep Eutectic Solvents (DESs). We trained XGBoost model to predict the cathode solubility in DESs. The Shapley additive explanation (SHAP) method was used to qualify the importance of each property. Next, we developed a CGAN model to identify promising DESs with excellent predictions and experimental results. Given the desired DESs leaching cathodes and the DESs leaching metal oxides, the model can generate a list of possible DES properties, which facilitates the discovery of novel promising DES properties.
The link of this project is: https://pubs.rsc.org/en/content/articlelanding/2024/gc/d4gc01418a/unauth

Project structure:
cgan.py: initializes and trains the CGAN model, incorporating both attention and residual mechanisms to enhance learning.

data_process.py: Prepares the dataset by adding two additional columns representing the hydrogen bond acceptor and hydrogen bond donor for each data entry, essential for modeling DES properties.

filter.py: run the model’s corresponding code in model.py. It helps the user to train the models quickly and easily.

fun.py: Contains utility functions that calculate the similarity between molecules based on their structural features, aiding in the comparison and evaluation of DES candidates.

model.py: Defines classes for four different models, allowing for comprehensive model experimentation.

predict.py: Outputs the best performance metrics for each model, summarizing their effectiveness in predicting DES properties.

test.py: a test to utilize CGAN with attention and residual mechanism to generate descriptors, assess the similarity between the generated and real descriptors, and performs predictions.

train_test.py: prepare the training and testing datasets for machine learning models.
