# SmallMedLM: Fine-Tuning GPT-2 for Medical Data
## Overview
This project involves fine-tuning a small version of the GPT-2 model (distilgpt2) on a dataset of diseases and symptoms. The goal is to train a language model that can generate text related to medical conditions and their symptoms.

The project is implemented in a Google Colab notebook and covers data loading, preprocessing, model training, and evaluation. The final model is saved and can be used for generating medical-related text based on input queries.

## Contents
1. [Setup](#setup)
Data Preparation
Model Training
Generating Predictions
Usage
Acknowledgements

## Setup
The following Python packages are required for this project:

torch
torchtext
transformers
sentencepiece
pandas
tqdm
datasets
You can install these packages using the following command:

python
Copy code
!pip install torch torchtext transformers sentencepiece pandas tqdm datasets
Data Preparation
Load Data: The dataset used is QuyenAnhDE/Diseases_Symptoms, which contains information about various diseases and their symptoms.

Preprocess Data: The symptoms are formatted as comma-separated strings to make them easier to work with.

Create Dataset Class: A custom LanguageDataset class is defined for handling data in a format suitable for GPT-2 training.

Model Training
Model Selection: We use the distilgpt2 model, a smaller and faster version of GPT-2, for fine-tuning.

Training Loop: The training involves updating the model's weights using the CrossEntropyLoss function and the Adam optimizer. The training and validation losses are logged for each epoch.

Parameters:

Batch Size: 8
Learning Rate: 5e-4
Number of Epochs: 10
Device Configuration: The model training can run on GPU, MPS, or CPU depending on the available hardware.

Generating Predictions
Once the model is trained, you can use it to generate text based on input queries. For example, given the input string "Kidney Failure", the model generates related text.

python
Copy code
input_str = "Kidney Failure"
input_ids = tokenizer.encode(input_str, return_tensors='pt').to(device)

output = model.generate(
    input_ids,
    max_length=20,
    num_return_sequences=1,
    do_sample=True,
    top_k=8,
    top_p=0.95,
    temperature=0.5,
    repetition_penalty=1.2
)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
Usage
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/SmallMedLM.git
Open the Colab Notebook:

Upload the notebook to Google Colab.
Run each cell to execute the code.
Load and Use the Model:

Use the saved model file (SmallMedLM.pt) for generating predictions or further training.
