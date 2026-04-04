# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

class AiSalaryPredictor:

    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.results = {}

    def load_data(self):
        print(self.df.head())

    def model_evaluation(self, y_pred, y_test, model_name):

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store results
        self.results[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

        print(f"\n{model_name}")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")

    def fit_model(self, pipe, X_train, y_train, X_test):
        pipe.fit(X_train, y_train)
        return pipe.predict(X_test)

    def tune_model(self, grid, X_train, y_train, X_test):
        grid.fit(X_train, y_train)
        print("\nBest Parameters:", grid.best_params_)
        print("Best CV Score:", grid.best_score_)
        return grid.best_estimator_.predict(X_test)

    def splitting_data(self):
        X = self.df.drop(["salary_usd"], axis=1)
        y = self.df["salary_usd"]
        return train_test_split(X, y, train_size=0.8, random_state=42)

    def create_preprocessor(self, model_type="linear"):

        cat_cols = ['country', 'job_role', 'ai_specialization', 'industry', 'work_mode', 'education_required']

        nominal = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown='ignore'))
        ])

        ordinal1 = Pipeline([
            ("ordinal", OrdinalEncoder(categories=[['Entry', 'Mid', 'Senior', 'Lead']]))
        ])

        ordinal2 = Pipeline([
            ("ordinal", OrdinalEncoder(categories=[['Startup', 'Small', 'Medium', 'Large', 'Enterprise']]))
        ])

        transform = Pipeline([
            ("log", FunctionTransformer(np.log1p))
        ])

        num_cols = [
            'weekly_hours', 'hiring_difficulty_score', 'ai_adoption_score', 'economic_index',
            'offer_acceptance_rate', 'tax_rate_percent', 'skill_demand_score', 'automation_risk',
            'job_security_score', 'career_growth_score', 'work_life_balance_score',
            'promotion_speed', 'salary_percentile', 'employee_satisfaction'
        ]

        transformers = [
            ("onehot", nominal, cat_cols),
            ("ordinal1", ordinal1, ['experience_level']),
            ("ordinal2", ordinal2, ['company_size']),
            ("log", transform, ['bonus_usd'])
        ]

        if model_type == "linear":
            transformers.append(("scaler", StandardScaler(), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

        return ColumnTransformer(transformers)

    def linear_regression(self):

        print("\nLinear Regression")

        preprocessor = self.create_preprocessor("linear")

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", LinearRegression())
        ])

        X_train, X_test, y_train, y_test = self.splitting_data()

        y_pred = self.fit_model(pipe, X_train, y_train, X_test)

        self.model_evaluation(y_pred, y_test, "Linear Regression")

    def svm(self):

        print("\nSupport Vector Machine")

        preprocessor = self.create_preprocessor("linear")

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", LinearSVR())
        ])

        param_grid = {
            'model__C': [0.01, 0.1, 1],
            'model__epsilon': [0.01, 0.1, 0.5],
            'model__max_iter': [1000, 2000]
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(pipe, param_grid, cv=kf, scoring='r2', n_jobs=-1)

        X_train, X_test, y_train, y_test = self.splitting_data()

        # Before tuning
        y_pred = self.fit_model(pipe, X_train, y_train, X_test)
        self.model_evaluation(y_pred, y_test, "SVM (Before Tuning)")

        # After tuning
        y_pred = self.tune_model(grid, X_train, y_train, X_test)
        self.model_evaluation(y_pred, y_test, "SVM (After Tuning)")

    def decision_tree(self):

        print("\nDecision Tree")

        preprocessor = self.create_preprocessor("tree")

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", DecisionTreeRegressor(random_state=42))
        ])

        param_grid = {
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        grid = GridSearchCV(pipe, param_grid, cv=kf, scoring="r2", n_jobs=-1)

        X_train, X_test, y_train, y_test = self.splitting_data()

        # Before tuning
        y_pred = self.fit_model(pipe, X_train, y_train, X_test)
        self.model_evaluation(y_pred, y_test, "Decision Tree (Before Tuning)")

        # After tuning
        y_pred = self.tune_model(grid, X_train, y_train, X_test)
        self.model_evaluation(y_pred, y_test, "Decision Tree (After Tuning)")

    def random_forest(self):

        print("\nRandom Forest")

        preprocessor = self.create_preprocessor("tree")

        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
        ])

        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 3]
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        grid = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=10,
            cv=kf,
            scoring="r2",
            n_jobs=-1,
            random_state=42
        )

        X_train, X_test, y_train, y_test = self.splitting_data()

        # Before tuning
        y_pred = self.fit_model(pipe, X_train, y_train, X_test)
        self.model_evaluation(y_pred, y_test, "Random Forest (Before Tuning)")

        # After tuning
        y_pred = self.tune_model(grid, X_train, y_train, X_test)
        self.model_evaluation(y_pred, y_test, "Random Forest (After Tuning)")

    def compare_models(self):

        print("\nFinal Model Comparison\n")

        for model, metrics in self.results.items():
            print(f"\n{model}")
            print(f"  R2: {metrics['R2']:.4f}")
            print(f"  MAE: {metrics['MAE']:.2f}")
            print(f"  RMSE: {metrics['RMSE']:.2f}")

        # Best model
        best_model = max(self.results, key=lambda x: self.results[x]['R2'])
        print(f"\nBest Model: {best_model}")

if __name__ == "__main__":

    predictor = AiSalaryPredictor('Dataset/global_ai_jobs.csv')
    predictor.load_data()

    predictor.linear_regression()
    predictor.svm()
    predictor.decision_tree()
    predictor.random_forest()

    predictor.compare_models()