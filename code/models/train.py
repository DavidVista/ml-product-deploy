from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics._scorer import make_scorer
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve


def load_dataset(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filename, index_col=0)
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])

    X['TotalLateDays'] = X['NumberOfTime30-59DaysPastDueNotWorse'] + \
        X['NumberOfTime60-89DaysPastDueNotWorse'] + \
        X['NumberOfTimes90DaysLate']

    X['IncomeExpenseDifference'] = X['MonthlyIncome'] - X['MonthlyIncome'] * X['DebtRatio']
    X['90DaysLateLikelihood'] = (0.1 * X['NumberOfTime30-59DaysPastDueNotWorse'] +
                                 0.2 * X['NumberOfTime60-89DaysPastDueNotWorse'] +
                                 0.7 * X['NumberOfTimes90DaysLate'])

    X = X.drop(columns=['DebtRatio',
                        'MonthlyIncome',
                        'NumberOfTime30-59DaysPastDueNotWorse',
                        'NumberOfTime60-89DaysPastDueNotWorse',
                        'NumberOfTimes90DaysLate'])

    return X, y


def preprocess(X: np.array, train=True) -> np.array:
    if train:
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_preprocessed = scaler.fit_transform(X_imputed)

        with open('../../models/imputer.pkl', 'wb') as imputer_file:
            imputer = pickle.dump(imputer, imputer_file)
        with open('../../models/scaler.pkl', 'wb') as scaler_file:
            scaler = pickle.dump(scaler, scaler_file)

        return X_preprocessed

    with open('../../models/imputer.pkl', 'rb') as imputer_file:
        imputer = pickle.load(imputer_file)
    with open('../../models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    X_imputed = imputer.transform(X)
    X_preprocessed = scaler.transform(X_imputed)

    return X_preprocessed


def train_and_save(model: BaseEstimator, X, y):
    model.fit(X, y)

    with open('../../models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return model


def main(val=True, test=True):
    X, y = load_dataset('../datasets/cs-training.csv')

    # Undersampling

    np.random.seed(42)

    pos_class_index = y[y == 1].index
    neg_class_index = y[y == 0].index.to_numpy()

    np.random.shuffle(neg_class_index)

    selected_negative = pd.Index(neg_class_index[:len(pos_class_index)])

    undersample_index = pos_class_index.union(selected_negative)

    X = X.loc[undersample_index]
    y = y.loc[undersample_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)

    X_train_prep = preprocess(X_train)

    params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'subsample': 1.0,
        'max_depth': 4,
        'n_iter_no_change': 10,
        'validation_fraction': 0.2
    }

    model = GradientBoostingClassifier(**params, random_state=42, verbose=True)

    if val:
        print("Validation Started")
        cv = cross_validate(
            GradientBoostingClassifier(**params, random_state=42, verbose=False),
            X_train,
            y_train,
            scoring=make_scorer(roc_auc_score, greater_is_better=True)
        )

        print(f"Average ROC-AUC score over 5 folds: {np.mean(cv['test_score'])}")

    train_and_save(model, X_train_prep, y_train)

    if test:
        X_test_prep = preprocess(X_test, train=False)

        y_pred_proba = model.predict_proba(X_test_prep)

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])

        plt.plot(thresholds, precision[:-1], color='red', label='precision')
        plt.plot(thresholds, recall[:-1], color='blue', label='recall')

        plt.grid()
        plt.legend()
        plt.show()

        best_threshold = 0.4982

        y_pred = (y_pred_proba[:, 1] > best_threshold).astype(np.int32)

        print(
            (f"Acc.: {accuracy_score(y_test, y_pred)}, "
             f"Prec.: {precision_score(y_test, y_pred)}, "
             f"Rec.: {recall_score(y_test, y_pred)}, "
             f"ROC-AUC: {roc_auc_score(y_test, y_pred)}")
        )


def test():
    W, _ = load_dataset('../datasets/cs-test.csv')
    W_prep = preprocess(W)

    with open('../../models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

        z_pred = model.predict_proba(W_prep)

        output = pd.DataFrame({'id': W.index, 'Probability': z_pred[:, 1]})

        output.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main(val=False)
    # test()
