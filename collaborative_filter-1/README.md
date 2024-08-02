# Collaborative Filtering Recommendation System

## Overview
This project focuses on building a collaborative filtering recommendation system using PySpark and Google Cloud Dataproc. Collaborative filtering predicts the interests of a user by collecting preferences from many users.

## Problem Statement
The goal is to provide accurate movie recommendations by predicting user ratings for movies they haven't watched yet, based on historical user-movie interaction data.

## Scope
The project involves:
- Data preparation
- Model training using Alternating Least Squares (ALS) in PySpark
- Evaluation of the model's performance
- Deployment on Google Cloud Dataproc

## Table of Contents
1. [Setup the Environment](https://github.com/KadirbekSharau/BigData/blob/main/collaborative_filter-1/Kadirbek%20Sharau%208.pdf)


## Setup the Environment
1. Open Google Cloud Console.
2. Activate Cloud Shell.
3. Authenticate with Google Cloud Platform (GCP).
4. Create a Google Cloud Storage bucket to store data.

## Download Programs and Related Documentation
- Clone the repository: `git clone https://github.com/ASD-Are/Big_Data`
- Download the MovieLens dataset: [MovieLens 100k Dataset](https://files.grouplens.org/datasets/movielens/ml-100k/u.data)

## Process of Program Execution

### Data Preparation
1. **Create the `u.data` File**: 
   ```bash
   vim u.data
