---
editor_options: 
  markdown: 
    wrap: 72
---

# CV4E_unsupProj

This project is a result of the CV4E summer school, and is exploring the
possibillity of measuring species richness from camera trap images,
without prior labelling of species.

Table of contents

1.  Repo structure[#repo-structure]

2.  Running code

3.  Data sources

##1.Repo structure:

The structure of this repo is as follows:

-   Configs

    -   A yaml file containing configuration settings used for all
        models

-   data_setup

    -   Python scripts containing methods for data augmentation as well
        as choosing split of train, test and val datasets.
    -   `ListTrainTest.py` - creates lists of img paths in each
        train/test/val set and writes to a .txt file.

-   unsupProj

    -   The main project folder with python scripts:

    -   `dataset.py` setting dataset utility functions for cropped CT
        images

    -   `model.py` creating class functions for each model to be used

    -   `functions.py` utility functions for embedding prediction and
        cluster analysis

    -   `predict.py` utilising predict functions and models to create
        feature embeddings

    -   `evaluate.py` taking output from model predictions and running
        clustering analysis

3.  Data Sources

    Models used in this project:

    -   **PegNet** = ResNet50, pretrained on ImageNet, downloaded
        directly from PyTorch

    -   **SwavNet** = ResNet50, trained using an unsupervised method,
        also part of PyTorch hub.
        <https://github.com/facebookresearch/swav>

    -   **EmbNet** = A ResNet50, using weights from a model trained on
        camera trap data from the Masai Mara, using a contrastive loss
        function (SimCLR)
        <https://github.com/omipan/camera_traps_self_supervised>
