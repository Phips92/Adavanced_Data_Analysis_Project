
# Advanced Data Analysis Project

## Project Overview
This project focuses on analyzing emotions from text data, specifically tweets, using machine learning (ML) and natural language processing (NLP) techniques. The primary goal of this project is to investigate how well synthetic data generated by ChatGPT can represent emotions such as anger, fear, joy, love, sadness, and surprise compared to real-world data from the Kagel dataset. The analysis focuses on understanding the strengths and limitations of synthetic data in capturing emotional nuances and its effectiveness when used to train machine learning models for emotion recognition.
This research highlights the potential value of synthetic data as a substitute for real-world data, especially in scenarios where collecting and labeling real-world data is expensive and time-consuming.If synthetic data proves effective, it could serve as a valuable resource for testing hypotheses and accelerating research, not only in emotion recognition but also in other fields.

---

## Features
- Training models on synthetic and real datasets.
- Comparison of Bag-of-Words (BoW) and TF-IDF vectorization techniques.
- Use of topic modeling methods such as Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA).
- Cross-dataset evaluation to compare the performance of synthetic and real data.

---

## Technologies and Tools
The following Python libraries were used:
1. **pandas, numpy, matplotlib**: For data manipulation and visualization.
2. **nltk, re**: For natural language preprocessing (tokenization, stopword removal, lemmatization).
3. **scikit-learn**: For vectorization, model training, and evaluation.
4. **pytest**: For testing and validating data preprocessing steps.


---

## Installation and Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/Phips92/Advanced_Data_Analysis_Project.git
   cd Advanced_Data_Analysis_Project
   ```
2. Install the required Python libraries:
   ```bash
    missing yet
   ```

---

## Dataset Information
1. **Kagel Dataset**: "Emotions - Where Words Paint the Colors of Feelings" ([Kagel Link](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)).
2. **Synthetic Dataset**: Generated using ChatGPT, containing tweets written in American English, labeled with one of the six emotions (Equaly distributed).

---

## How to Use
Train models using scripts like Kagel_model.py, Synthetic_model.py, Kaggle_model2.py, or Whole_kaggle_model.py to analyze performance. Experiment with different feature extraction techniques such as Bag-of-Words (BoW), TF-IDF, LSA, or LDA, and evaluate models like Logistic Regression, SVM, and others for emotion recognition.

---

## Results
The analysis revealed that:
- Models trained on real Kagel data consistently outperformed those trained on synthetic data.
- Models trained on real Kagel data perform poorly when tested on synthetic data.
- The Bag-of-Words approach slightly outperformed TF-IDF in most scenarios.
- Topic modeling techniques (LSA, LDA) did not improve model performance.

---

## Contribution
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## License
This project is licensed under the GNU General Public License v3.0

---

## Author
Philip (Phips92)

For questions or feedback, feel free to contact me at [philipp92.mcguire@gmail.com].
