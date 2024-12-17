The goal of this experiment was to evaluate the performance of various models trained on synthetic data and Kaggle datasets to classify emotions. The findings clearly demonstrate that synthetic data generated using ChatGPT is significantly less effective in training accurate emotion classification models compared to real data. Furthermore, models trained on Kaggle data were trained on a reduced sample of 5,000 entries, equally distributed across classes, which provides a fair comparison and highlights the limitations of synthetic data.




Results:

Synthetic_model.py

_______Models trained with synthetic data and tested on another syntethic dataset______

model with BoW
Accuracy: 0.1704
              precision    recall  f1-score   support

           0       0.17      0.84      0.28       830
           1       0.17      0.08      0.11       834
           2       0.20      0.04      0.06       835
           3       0.18      0.03      0.05       832
           4       0.20      0.02      0.04       832
           5       0.14      0.01      0.03       837

    accuracy                           0.17      5000
   macro avg       0.18      0.17      0.09      5000
weighted avg       0.18      0.17      0.09      5000

 -Result: Class 0 dominates the results with the highest recall (0.84), while other classes show very low recall and precision. This suggests the model heavily biases predictions toward class 0.



model with TF-IDF
Accuracy: 0.171
              precision    recall  f1-score   support

           0       0.17      0.78      0.28       830
           1       0.17      0.08      0.11       834
           2       0.19      0.06      0.09       835
           3       0.16      0.04      0.07       832
           4       0.18      0.05      0.08       832
           5       0.15      0.02      0.04       837

    accuracy                           0.17      5000
   macro avg       0.17      0.17      0.11      5000
weighted avg       0.17      0.17      0.11      5000

 -Result: Similar to BoW, class 0 achieves high recall (0.78), while other classes remain poorly predicted, highlighting the lack of generalization in synthetic data.


LSA with TF-IDF
Accuracy: 0.171
              precision    recall  f1-score   support

           0       0.17      0.78      0.28       830
           1       0.17      0.08      0.11       834
           2       0.19      0.06      0.09       835
           3       0.16      0.04      0.07       832
           4       0.18      0.05      0.08       832
           5       0.15      0.02      0.04       837

    accuracy                           0.17      5000
   macro avg       0.17      0.17      0.11      5000
weighted avg       0.17      0.17      0.11      5000

 -Result: The performance mirrors the TF-IDF model, with class 0 receiving higher recall at the cost of other classes. This indicates that dimensionality reduction via LSA did not improve predictions.



LSA with BoW
Accuracy: 0.1704
              precision    recall  f1-score   support

           0       0.17      0.84      0.28       830
           1       0.17      0.08      0.11       834
           2       0.20      0.04      0.06       835
           3       0.18      0.03      0.05       832
           4       0.20      0.02      0.04       832
           5       0.14      0.01      0.03       837

    accuracy                           0.17      5000
   macro avg       0.18      0.17      0.09      5000
weighted avg       0.18      0.17      0.09      5000

 -Result: Results are nearly identical to BoW, where the model overly focuses on class 0, showing poor balance across classes.

LDA with BoW
Accuracy: 0.1632
              precision    recall  f1-score   support

           0       0.17      0.29      0.21       830
           1       0.18      0.06      0.10       834
           2       0.17      0.12      0.14       835
           3       0.16      0.34      0.21       832
           4       0.17      0.08      0.11       832
           5       0.15      0.09      0.11       837

    accuracy                           0.16      5000
   macro avg       0.17      0.16      0.15      5000
weighted avg       0.17      0.16      0.15      5000

 -Result: Unlike the previous methods, LDA shows slightly improved balance across classes but with lower overall accuracy. This result demonstrates LDA's difficulty in capturing meaningful patterns from synthetic data.



_______Models trained with synthetic data and tested on kaggle dataset______

model with BoW
Accuracy: 0.30347473303119654
              precision    recall  f1-score   support

           0       0.30      0.86      0.45    121187
           1       0.46      0.10      0.17    141067
           2       0.16      0.06      0.09     34554
           3       0.25      0.06      0.09     57317
           4       0.28      0.05      0.08     47712
           5       0.03      0.02      0.02     14972

    accuracy                           0.30    416809
   macro avg       0.25      0.19      0.15    416809
weighted avg       0.33      0.30      0.22    416809

 -Result: Class 0 dominates predictions with high recall (0.86). Other classes, particularly class 1, achieve minimal recall, showing the model's bias and failure to generalize to real data.


model with TF-IDF
Accuracy: 0.29629878433527107
              precision    recall  f1-score   support

           0       0.31      0.79      0.44    121187
           1       0.48      0.11      0.17    141067
           2       0.13      0.07      0.09     34554
           3       0.27      0.09      0.14     57317
           4       0.20      0.08      0.11     47712
           5       0.07      0.06      0.07     14972

    accuracy                           0.30    416809
   macro avg       0.24      0.20      0.17    416809
weighted avg       0.32      0.30      0.23    416809

 -Result: Performance is very similar to BoW. Class 0 still dominates predictions, while recall and precision for other classes remain low


LSA with TF-IDF
Accuracy: 0.29629878433527107
              precision    recall  f1-score   support

           0       0.31      0.79      0.44    121187
           1       0.48      0.11      0.17    141067
           2       0.13      0.07      0.09     34554
           3       0.27      0.09      0.14     57317
           4       0.20      0.08      0.11     47712
           5       0.07      0.06      0.07     14972

    accuracy                           0.30    416809
   macro avg       0.24      0.20      0.17    416809
weighted avg       0.32      0.30      0.23    416809

 -Result: LSA does not improve performance, as the model continues to favor class 0. This emphasizes the weak representation of emotional patterns in synthetic data.


LSA with BoW
Accuracy: 0.30347473303119654
              precision    recall  f1-score   support

           0       0.30      0.86      0.45    121187
           1       0.46      0.10      0.17    141067
           2       0.16      0.06      0.09     34554
           3       0.25      0.06      0.09     57317
           4       0.28      0.05      0.08     47712
           5       0.03      0.02      0.02     14972

    accuracy                           0.30    416809
   macro avg       0.25      0.19      0.15    416809
weighted avg       0.33      0.30      0.22    416809

 -Result: Similar to the other models, accuracy is primarily driven by class 0's recall, highlighting the model's inability to generalize effectively.

LDA with BoW
Accuracy: 0.19539645257180147
              precision    recall  f1-score   support

           0       0.30      0.31      0.31    121187
           1       0.42      0.07      0.11    141067
           2       0.09      0.13      0.11     34554
           3       0.15      0.45      0.23     57317
           4       0.11      0.05      0.07     47712
           5       0.04      0.08      0.06     14972

    accuracy                           0.20    416809
   macro avg       0.19      0.18      0.15    416809
weighted avg       0.27      0.20      0.18    416809

 -Result: LDA demonstrates the poorest performance, although class 3 achieves better recall. Overall, the model fails to adapt to real-world data.


_________________________________________
combine_dataset.py

_______Models trained with combined synthetic data sets and tested on kaggle dataset______

model with BoW
Accuracy: 0.2932902120635591
              precision    recall  f1-score   support

           0       0.31      0.65      0.42    121187
           1       0.43      0.18      0.26    141067
           2       0.13      0.07      0.09     34554
           3       0.22      0.15      0.18     57317
           4       0.17      0.13      0.14     47712
           5       0.12      0.08      0.09     14972

    accuracy                           0.29    416809
   macro avg       0.23      0.21      0.20    416809
weighted avg       0.30      0.29      0.26    416809

 -Result: The model shows slightly more balanced performance across classes, but accuracy remains low overall. Class 0 continues to dominate, showing the model's bias.

Model with TF-IDF
Accuracy: 0.27671187522342366
              precision    recall  f1-score   support

           0       0.31      0.57      0.40    121187
           1       0.45      0.16      0.23    141067
           2       0.12      0.08      0.10     34554
           3       0.23      0.18      0.20     57317
           4       0.14      0.19      0.16     47712
           5       0.15      0.16      0.15     14972

    accuracy                           0.28    416809
   macro avg       0.23      0.22      0.21    416809
weighted avg       0.31      0.28      0.26    416809

 -Result: Similar to BoW, the TF-IDF model achieves marginal performance gains across classes. However, overall accuracy and balance remain poor.

_________________________________________
Kagel_model.py

___Models training with Kaggle dataset and tested on synthetic dataset___

model with BoW
Accuracy: 0.3196
              precision    recall  f1-score   support

           0       0.27      0.30      0.28       831
           1       0.50      0.30      0.38       830
           2       0.00      0.00      0.00       841
           3       0.25      0.90      0.39       839
           4       1.00      0.31      0.47       830
           5       0.52      0.10      0.17       829

    accuracy                           0.32      5000
   macro avg       0.42      0.32      0.28      5000
weighted avg       0.42      0.32      0.28      5000

 -Result: The model achieves improved accuracy compared to synthetic-trained models, indicating that real data is more informative. However, predictions still suffer from class imbalance, as class 3 achieves very high recall but fails to generalize to class 2, which receives no predictions. 


model with TF-IDF
Accuracy: 0.3976
              precision    recall  f1-score   support

           0       0.12      0.19      0.15       831
           1       0.31      0.51      0.39       830
           2       0.57      0.40      0.47       841
           3       0.39      0.39      0.39       839
           4       0.80      0.80      0.80       830
           5       0.52      0.10      0.17       829

    accuracy                           0.40      5000
   macro avg       0.45      0.40      0.39      5000
weighted avg       0.45      0.40      0.40      5000

 -Result: This model demonstrates improved generalization compared to BoW, with better recall and precision across several classes, particularly class 4 (0.80).


LSA with Tf-IDF
Accuracy: 0.4296
              precision    recall  f1-score   support

           0       0.26      0.28      0.27       831
           1       0.33      0.51      0.40       830
           2       0.56      0.50      0.53       841
           3       0.39      0.39      0.39       839
           4       0.66      0.80      0.72       830
           5       0.33      0.10      0.16       829

    accuracy                           0.43      5000
   macro avg       0.42      0.43      0.41      5000
weighted avg       0.42      0.43      0.41      5000

 -Result: LSA improves performance slightly compared to TF-IDF, achieving a more balanced performance across multiple classes, particularly classes 2, 3, and 4.


LSA with BoW
Accuracy: 0.3632
              precision    recall  f1-score   support

           0       0.32      0.48      0.39       831
           1       0.35      0.40      0.38       830
           2       0.00      0.00      0.00       841
           3       0.27      0.69      0.39       839
           4       1.00      0.50      0.67       830
           5       0.52      0.10      0.17       829

    accuracy                           0.36      5000
   macro avg       0.41      0.36      0.33      5000
weighted avg       0.41      0.36      0.33      5000

 -Result: The model performs reasonably well but fails to generalize to class 2, which receives no predictions. Class 4 achieves higher precision asbefore with BoW.

LDA with BoW
Accuracy: 0.1188
              precision    recall  f1-score   support

           0       0.11      0.11      0.11       831
           1       0.06      0.09      0.08       830
           2       0.11      0.10      0.11       841
           3       0.20      0.31      0.25       839
           4       0.00      0.00      0.00       830
           5       0.25      0.10      0.14       829

    accuracy                           0.12      5000
   macro avg       0.12      0.12      0.11      5000
weighted avg       0.12      0.12      0.11      5000

 -Result: This model shows the weakest performance, with low precision and recall across all classes, highlighting LDA's poor performance on synthetic data.

___________________________________________

Kagel_model2.py

___Models training with Kaggle dataset train/tst split 0.15 and tested on kaggle dataset___

Bow(Kaggle train/test split, test_size=0.95)
Accuracy: 0.885240395380826
              precision    recall  f1-score   support

           0       0.94      0.94      0.94     18178
           1       0.91      0.91      0.91     21160
           2       0.75      0.75      0.75      5183
           3       0.89      0.89      0.89      8598
           4       0.83      0.83      0.83      7157
           5       0.69      0.70      0.69      2246

    accuracy                           0.89     62522
   macro avg       0.83      0.83      0.83     62522
weighted avg       0.89      0.89      0.89     62522



TF-IDF (Kaggle train/test split, test_size=0.15)
Accuracy: 0.8925978055724385
              precision    recall  f1-score   support

           0       0.93      0.94      0.94     18178
           1       0.91      0.93      0.92     21160
           2       0.79      0.75      0.77      5183
           3       0.89      0.89      0.89      8598
           4       0.84      0.83      0.84      7157
           5       0.74      0.70      0.72      2246

    accuracy                           0.89     62522
   macro avg       0.85      0.84      0.85     62522
weighted avg       0.89      0.89      0.89     62522



Summary

    Synthetic Data vs. Real Data:
        Models trained on synthetic data (ChatGPT-generated) perform poorly, with accuracy around 17%-30%. These models heavily bias predictions toward dominant classes (e.g., class 0).
        In contrast, models trained on Kaggle data with a train-test split of 0.15 achieve accuracies of 88.5% (BoW) and 89.3% (TF-IDF).

    Performance on Real Data:
        BoW and TF-IDF models trained on Kaggle data perform remarkably well, achieving balanced precision and recall across all classes. Even smaller classes like class 5 reach reasonable performance.

    Generalization Issues:
        Models trained on synthetic data fail to generalize to real datasets, emphasizing the limitations of ChatGPT-generated data in capturing emotional nuances.

    Model Comparison:
        LDA fails to provide meaningful improvements in performance when trained on synthetic data.
        LSA slightly improves performance, especially when combined with TF-IDF and trained on real data.
        TF-IDF outperforms BoW when trained on real data but does not show the same advantage when trained on synthetic data.


Conclusion

The results clearly highlight the superiority of real data (Kaggle dataset) over synthetic data for emotion classification. Models trained on real data achieve nearly 90% accuracy, while those trained on synthetic data perform poorly. This confirms that ChatGPT struggles to generate high-quality, emotionally diverse datasets, further supporting the notion that ChatGPT is limited in its ability to understand or represent emotions effectively.

