# Reproducing GraphRec for Top-K Evaluation: A Case Study on Diversity Metrics

## Introduction

This repository is an adaptation and extension of the original [GraphRec implementation](https://github.com/wenqifan03/GraphRec-WWW19) by Fan et al., presented in their paper "Graph Neural Networks for Social Recommendation." Our work, "Reproducing GraphRec for Top-K Evaluation: A Case Study on Diversity Metrics," builds upon the foundational concepts of GraphRec and delves into its reproducibility, particularly focusing on top-k evaluation metrics.

In the dynamic field of recommender systems, the GraphRec model, utilizing graph neural networks (GNNs) for social recommendation, has emerged as a significant approach. Our research not only reproduces but also extends the evaluation of GraphRec, highlighting several challenges, especially when shifting from rating prediction to a top-k offline evaluation setting. This work underscores the critical need for clear, detailed preprocessing documentation and brings to light the complexities involved in achieving reproducible results, even when the original codebase is accessible.

Our study is grounded in the context of the original GraphRec model, and we recommend referring to the [original GraphRec paper](https://dl.acm.org/doi/10.1145/3308558.3313417) for foundational knowledge:

> Fan, W., Ma, Y., Li, Q., He, Y., Zhao, E., Tang, J., & Yin, D. (2019, May). Graph Neural Networks for Social Recommendation. In The World Wide Web Conference (pp. 417-426).
### Abstract

_In the dynamic field of recommender systems, the GraphRec model has emerged as a popular approach. This paper details our efforts to reproduce and extend GraphRec's evaluation, revealing several challenges. We emphasize the need for detailed preprocessing documentation, discuss our transition to top-k evaluation setup, and point out the often-overlooked challenges in achieving reproducibility. Our results focus on recommendation diversity, showcasing the importance of top-k evaluations. Through this work, we aim to underscore the challenges of reproducibility in social recommendation systems._

## Research Questions

Our research revolves around three central questions:

1. **RQ1:** How does recommender system documentation clarity and availability influence the reproducibility of GraphRec?
2. **RQ2:** What considerations arise when transitioning GraphRec from rating prediction to top-k evaluations?
3. **RQ3:** How can best practices be improved for consistent and reproducible evaluations?

By exploring these questions, we seek to illuminate the often intricate process of model reproduction, particularly within the realm of social recommendation systems.

## Contributions

This study makes several contributions:

- An in-depth analysis of GraphRec's data preprocessing challenges, particularly with the Epinions and Ciao datasets.
- Insights on adapting GraphRec for top-k recommendation evaluation, highlighting crucial decisions and their implications.
- Best practices derived from our findings, intended to guide consistent and reproducible evaluations in future social recommendation research.

## Repository Structure

This repository is organized as follows:

- `data/`: Directory for storing raw and processed data.
- `notebooks/`: Jupyter notebooks for various data explorations and analyses.
- `elliot_config/`: Configuration files for baseline evaluations using the Elliot framework.

### Main Files

- `download_datasets.py`: Script to download the Ciao and Epinions datasets from the specified source. The datasets are stored in `./data/raw`.
- `preprocess_datasets.py`: Script for preprocessing the datasets. It includes various parameters for cleaning and structuring the data suitable for GraphRec and Elliot. Processed data is stored in `./data/processed/graphrec/` and `./data/processed/elliot/`.
- `run_GraphRec_example.py`: An executable script that runs the original GraphRec approach for rating prediction. It includes the preprocessing pipeline and various parameters for dataset processing and model training.
- `run_GraphRec_example_topk_BCE.py`: This script is used for evaluating the modified GraphRec approach using Binary Cross Entropy (BCE) in a top-k evaluation setting. It includes parameters for training and evaluation.
- `run_ciao.sh` and `run_epinions.sh`: Shell scripts providing examples on how to execute `run_GraphRec_example.py` with different parameter combinations on the Ciao and Epinions datasets, respectively.
- `./notebooks/data_exploration.py`: A Jupyter notebook containing data exploration for toy_dataset, Epinions, and Ciao datasets.

### Configuration Files

- `./elliot_config/`: This directory contains various configuration files used for baseline evaluations within the Elliot framework. More details about Elliot can be found [here](https://elliot.readthedocs.io/en/latest/).

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of [Python](https://www.python.org/).
- You have a Windows/Linux/Mac machine capable of running Python 3.
- You have installed [PyTorch](https://pytorch.org/). If you plan to run the model on a GPU, make sure to follow PyTorch’s instructions for [CUDA support](https://pytorch.org/get-started/locally/).

## Setting Up and Running the Project

This project is designed to be straightforward to set up and run. Below are the steps you need to follow to get everything ready and to initiate the model training:

1. **Clone the repository:**

    Start by cloning the repository to your local machine using the following command:

    ```bash
    git clone https://anonymous.4open.science/r/GraphRec-ECIR24/
    ```

    Don't forget to replace `your_username` and `your_project_name` with your GitHub username and your repository's name, respectively.

2. **Navigate to the project directory:**

    Change directories to enter the main project folder:

    ```bash
    cd your_project_name
    ```

3. **Download the datasets:**

    Run the `download_datasets.py` script to fetch the Ciao and Epinions datasets. They will be stored in the appropriate `data` subdirectory.

    ```bash
    python download_datasets.py
    ```

4. **Preprocess the datasets:**

    Once you've downloaded the datasets, run the `preprocess_datasets.py` script to prepare the data for training. This script cleans and structures the data, making it suitable for input into the GraphRec model.

    ```bash
    python preprocess_datasets.py
    ```

    This script includes several parameters that you can customize based on your requirements. Refer to the script's documentation for more details.

5. **Model Training:**

    After preprocessing the data, you can start the model training process. You can use the provided shell scripts or Python scripts (e.g., `run_GraphRec_example.py` or `run_GraphRec_example_topk_BCE.py`) to train the GraphRec model with your desired parameters.

    ```bash
    python run_GraphRec_example.py  # or any other relevant script
    ```

    If you prefer to run the model on a GPU, ensure you have the necessary CUDA support as required by PyTorch. The model can also run on a CPU if you don't have GPU support.

6. **Evaluation:**

    After training, you can evaluate the model's performance using various metrics. The necessary scripts and notebooks in the repository will guide you through this process.

Please refer to the individual scripts' documentation within the repository for more detailed information on their usage and the various parameters you can adjust.
