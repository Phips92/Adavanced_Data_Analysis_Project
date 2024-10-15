from sklearn.metrics import classification_report, accuracy_score

class ModelEvaluator:

    def evaluate(self, model, X_test, y_test):

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        return report, accuracy
