# Smoking Trajectory Project

This project is an approach using machine learning to reproduce the results of the paper by Biocard and Jusot, 'Milieu d’origine, situation sociale et parcours tabagique en France.' The aim of this project is to provide an original contribution to this research topic through the use of neural networks.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Description](#data-description)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Usage](#usage)
6. [Citation](#citation)

## Project Overview

In this project, we aim to study the smoking trajectories of individuals in France and how they are influenced by their social origin and situation. We use the data from the ESPS survey conducted by the IRDES<https://www.irdes.fr/recherche/enquetes/esps-enquete-sur-la-sante-et-la-protection-sociale/actualites.html> in 2010, which is a cross-sectional survey that was conducted in France. The survey includes information on the health, lifestyle, and social situation of a representative sample of the French population.

## Data Description

The dataset used in this project includes 15,643 individuals aged 25-64 years and contains the following variables:

- **mere\_pcs**: Mother's profession and socio-professional category
- **pere\_pcs**: Father's profession and socio-professional category
- **mere\_etude**: Mother's level of education
- **pere\_etude**: Father's level of education
- **fumfoy1**: Father's smoking status
- **fumfoy2**: Mother's smoking status
- **sexe**: Gender of the individual
- **age**: Age of the individual
- **nivetu**: Level of education of the individual
- **pcs**: Profession and socio-professional category of the individual
- **fume**: Smoking status of the individual
- **afume**: Whether the individual has ever smoked
- **anfume**: Number of years since the individual started smoking
- **nbanfum**: Number of years the individual has smoked for
- **aarret**: Age at which the individual stopped smoking
- **anais**: Year of birth of the individual
- **quintuc**: Quintile of income
- **pond**: Weight of the individual in the survey
- **pers**: Number of people in the household
- **total**: Total income of the household
- **heberqgd**: Whether the individual experienced poverty during childhood

The dataset can be accessed from the following link: <https://www.irdes.fr/recherche/enquetes/esps-enquete-sur-la-sante-et-la-protection-sociale/actualites.html>

## Model Architecture

We have implemented three different neural network models in this project, which are as follows:

1. **Simple Model**: This is the simplest model.
2. **One-hot Model**: This model is a bit more complex than the simple model and includes one-hot encoding for the categorical variables. 
3. 3. **Age-visu Model**: This is the most complex and performant model in this project. It includes three separate models, each with a visible layer for the age variable, which allows the model to learn the influence of age on the smoking trajectories. The three models are as follows:
	* A model to predict whether the individual smokes or not, which achieves an accuracy of 90% on the test set.
	* A model to predict the age of smoking initiation, which predicts the age within 0.5 years of the actual age.
	* A model to predict the age of smoking cessation, which predicts the age within 0.5 years of the actual age.

## Results

The results of the Age-visu Model can be accessed from this website (https://smokingtrajectory.streamlit.app/).The model achieves an accuracy of 94% on the test set for predicting whether an individual smokes or not, which is a significant improvement compared to the simple and one-hot models. The visible layer for the age variable allows the model to learn the influence of age on the smoking trajectories and provides insights into the smoking behavior of individuals in France.

## Usage

To use this project, you can clone the repository and run the code in a Python environment. The code for each model is in a separate file, which are as follows:

-Simplemodel.py
-onhotvisu.py
-agevisu.py

Note that running the code may take some time, especially for the Age-visu Model, so please be patient.

## Citations 

If you use this project in your research, please cite me as follows:

-Name: Maury
-First Name: Octavien

