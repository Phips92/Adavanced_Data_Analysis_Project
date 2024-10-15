from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

class ModelTrainer:


    def __init__(self):

        self.models = {
            "logistic_regression": LogisticRegression(),
            "random_forest": RandomForestClassifier(),
            "svm": SVC(),
            "naive_bayes": MultinomialNB(),
            "gradient_boosting": GradientBoostingClassifier()
        }
    
    def train(self, X, y, model_type="logistic_regression"):

        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not supported!")
        
        model = self.models[model_type]
        model.fit(X, y)
        return model
