# Natural Language Processing Project (ITNPA4-34)

This repository contains the complete code, datasets, and deployment files for the Natural Language Processing Project. The project demonstrates foundational text mining, text preprocessing, machine learning classification, text summarization, and topic modeling using Python.

## Repository Structure

*   **data/**: Contains the original datasets used for the analysis. This includes the research abstracts JSON file and the folder containing the research corpus text files.
*   **notebooks/**: Contains the Python scripts that showcase the analytics flow for Questions 1 to 4. 
*   **deployment/**: Contains the Flask web application for the abstract classifier. Please refer to the README file inside this folder for specific deployment instructions.
*   **documentation/**: Contains the final project report detailing the methodology, results, and interpretations.

## Setup Instructions

To run the analysis scripts locally, you need to have Python installed along with the required libraries. You can install the core dependencies using the following command:

pip install pandas numpy scikit-learn nltk sumy wordcloud matplotlib seaborn pyLDAvis

## Running the Notebooks

The `notebooks` folder contains scripts for each question. 
*   Q1 handles data loading, preprocessing, and the baseline model.
*   Q2 handles advanced models, performance enhancement, and prepares the model for deployment.
*   Q3 handles text summarization and visualization.
*   Q4 handles Latent Dirichlet Allocation topic modeling and the pyLDAvis interactive map.

Navigate to the specific script you want to run and execute it. Ensure the file paths for the datasets in the scripts match your local environment.
