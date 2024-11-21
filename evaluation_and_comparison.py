from sklearn.metrics import classification_report, accuracy_score

class ModelEvaluator:
    """
    A class for evaluating ML models. It calculates evaluation metrics
    such as classification reports and accuracy scores for predictions made by a given model.
    
    Methods:
        evaluate(model, X_test, y_test): Evaluates a trained model on test data and
                                         returns the classification report and accuracy score.
    """

    def evaluate(self, model, X_test, y_test):

        y_pred = model.predict(X_test)                  # Predict labels for the test data
        report = classification_report(y_test, y_pred)  # Generate classification report
        accuracy = accuracy_score(y_test, y_pred)       # Calculate accuracy score
        
        return report, accuracy
