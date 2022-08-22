In this project, we tried to classify face images into five different categories: **['Heart', 'Oblong', 'Oval', 'Round', 'Square']**

Download dataset from https://www.kaggle.com/datasets/niten19/face-shape-dataset

Two different approaches are applied in order to classify face images. 

- In the first approach, we tried to extract features. Code available in `feature_extractor.py` and extracted features can be found in `feature_test.csv` and `feature_train.csv`. 
  - Applying **Dense Neural Network** (`dnn.py`) on the extracted features led to **64% accuracy**. 
  - Applying **Gradient Boosting** algorithm led to **64% accuracy**.
  
    
- In the second approach, we used **InceptionV3** for feature extraction. Results are available in `Inception Result` folder. Other measures for `inception.py` are reported below:
  - Accuracy: **79.6%**
  - Precision: **80.5%**
  - Recall: **79.6%**


