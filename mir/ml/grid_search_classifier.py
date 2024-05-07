import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mir.ml.grid_search_parameters import GRID_SEARCH_PARAMS
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator
from dataclasses import dataclass


@dataclass
class CustomMLModel:
    model_name: str
    model_object: BaseEstimator
    model_hyperparams: dict[str, list]


class GridSearchClassifier:
    """
    A class for classical ML models evaluation
    Uses grid search for hyperparameters optimization within 5-fold cross validation
    """

    def __init__(self,
                 used_models=['svm', 'ab', 'knn', 'rfc', 'xgboost', 'mlpclassifier'],
                 params_dict=None,
                 random_state=42,
                 custom_models: list[CustomMLModel] = None):
        """
        initializing function, nothing interesting
        :param used_models: the models which can be assessed; you can give any subset of the default set as the input, \
        but you should change the code if you want the new one to be added
        :param params_dict: the hyperparameters to be considered; by default GRID_SEARCH_PARAMS dict is used
        :param custom_models: you can pass your own models in addition to default ones by creating CustomMLModel objects
        """
        self.random_state = random_state
        self.best_params = None
        self.scores = None
        self.best_clfs = None
        self.best_model_name = None
        self.used_models = used_models
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        self.custom_models = None
        if custom_models is not None:
            self.custom_models = {}
            for model in custom_models:
                self.custom_models[model.model_name] = model

        self.params = self.get_parameters(self.used_models, params_dict)

    def get_model(self, model):
        '''
        Get the model from six state-of-the-art machine learning models.
        '''
        if model == 'svm':
            from sklearn.svm import SVC
            return SVC()
        elif model == 'ab':
            from sklearn.ensemble import AdaBoostClassifier
            return AdaBoostClassifier()
        elif model == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier()
        elif model == 'rfc':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()
        elif model == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier()
        elif model == 'mlpclassifier':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier()
        elif model in self.custom_models:
            return self.custom_models[model].model_object
        else:
            raise RuntimeError('Unknown classifier')


    def get_parameters(self, models: list = None, params_object: dict = None):
        """
        a fucntion which makes a hyperparameter dictionary
        :param models: a list of which models to consider
        :param params_object: a dictionary with parameters to consider or None if you want to use default
        :return: the dictionary of parameters
        """
        parameters_to_look_through = GRID_SEARCH_PARAMS if params_object is None else params_object
        if models is not None:
            parameters_to_look_through = {x: y for x, y in parameters_to_look_through.items() if x in models}
        if self.custom_models is not None:
            for model in self.custom_models:
                parameters_to_look_through[model] = self.custom_models[model].model_hyperparams

        for model in parameters_to_look_through:
            if 'random_state' in self.get_model(model).get_params():
                parameters_to_look_through[model]['random_state'] = [self.random_state]
        return parameters_to_look_through

    def specify_data_and_split(self, df, y, splitting_method=0.2, drop_columns=None):
        """
        a method for data passing; you should give a processed df as an input;
        it might contain the y column inside -- in this case you should pass this column name into `y` param
        the splitting method might be random (train test split from sklearn) or can be function based on initial df
        :param df: a pd.DataFrame of numeric features
        :param y: either a series of values or a string naming the column from `df`
        :param splitting_method: either a float value (piece of df to be taken out for validation) or a function \
        which should split the data (example function: lambda x: x.batch == 'test_batch')
        :param drop_columns: a list of columns which should be dropped from the initial data
        """
        X = df.reset_index(drop=True)
        if isinstance(y, str):
            y, X = X[y], X.drop(columns=[y])
        else:
            y = y.reset_index(drop=True)
        X_data = X if drop_columns is None else X.drop(columns=drop_columns)
        X_data = pd.DataFrame(data=StandardScaler().fit_transform(X_data), columns=X_data.columns)
        if isinstance(splitting_method, float):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_data, y, test_size=splitting_method, random_state=self.random_state)
        else:
            function_res = X.apply(splitting_method, axis=1)
            self.X_train, self.y_train = X_data[function_res], y[function_res]
            self.X_test, self.y_test = X_data[~function_res], y[~function_res]

    def evaluate_models(self, scoring_function='f1', n_folds=5, debug=False):
        """
        the models evaluation function
        :param scoring_function: should be one of sklearn.metrics function names (can be 'accuracy'/'f1'/'precision' \
        and so on)
        :param debug: whether to print the debug info or not
        :param n_folds: number of folds to split the data into for cross validation
        :return: a dict of best clfs for each model type and the best model name
        """
        skf = StratifiedKFold(n_splits=n_folds, random_state=self.random_state, shuffle=True)
        self.best_clfs = {}
        self.scores, self.best_params = [], []
        model_names = list(self.params.keys())

        for name, param in self.params.items():
            if debug:
                print(f'Started evaluating {name}')
            model = self.get_model(name)

            clf = GridSearchCV(model, param, scoring=scoring_function, cv=skf, n_jobs=-1)
            clf.fit(self.X_train, self.y_train)

            self.scores.append(f1_score(self.y_test, clf.predict(self.X_test)))
            self.best_params.append(clf.best_params_)
            if debug:
                print(f'Best params for {name}:', clf.best_params_)
                print('Test f1-score for the best model %.2f' % f1_score(self.y_test, clf.predict(self.X_test)))
                print()

            self.best_clfs[name] = clf.best_estimator_

        print(pd.DataFrame({'classifier': model_names,
                            'f1-score': self.scores
                            }).set_index('classifier').T)
        best_model_idx = max(range(len(self.scores)), key=lambda i: self.scores[i])
        self.best_model_name = model_names[best_model_idx]
        print(f'Best model is {self.best_model_name} with params: {self.best_params[best_model_idx]}')

        return self.best_clfs, self.best_model_name

    def plot_curves_for_clf(self, roc_ax=None, prrec_ax=None, name=None):
        """
        creates a roc/precision-recall curve
        if you don't specify both axes -- they are created;
        if you do only specify one of the axes -- one plot will be made
        :param roc_ax: the axes to plot the roc curve on;  will be created if not specified
        :param prrec_ax: the axes to plot the precision-recall curve on;  will be created if not specified
        :param name: which classifier to make plots for; if not specified the classifier with best evaluation metrics \
        is chosen
        """
        if name is None:
            name = self.best_model_name
        if roc_ax is None and prrec_ax is None:
            fig, (prrec_ax, roc_ax) = plt.subplots(1, 2, figsize=(8, 4))
        if prrec_ax is not None:
            PrecisionRecallDisplay.from_estimator(
                self.best_clfs[name], self.X_test,
                self.y_test, name=name, ax=prrec_ax
            )
            prrec_ax.set_ylim(0, 1)
        if roc_ax is not None:
            RocCurveDisplay.from_estimator(
                self.best_clfs[name], self.X_test,
                self.y_test, name=name, ax=roc_ax
            )

    def plot_curve_for_all_clfs(self, type='prrec', ax=None):
        """
        creates a precision-recall or roc curve for all the evaluated classifiers
        if you don't specify axes -- it is created;
        :param type: which plot should be made; precision-recall if type=='prrec' else roc curve
        :param ax: the axes to plot on; will be created if not specified
        """
        if ax is None:
            fig, ax = plt.subplots()
        for name in self.best_clfs.keys():
            self.plot_curves_for_clf(roc_ax=None if type == 'prrec' else ax,
                                     prrec_ax=None if type != 'prrec' else ax,
                                     name=name)

    def plot_confusion_matrix_for_clf(self, ax=None, name=None):
        """
        creates a confusion matrix for the model specified in `name` param
        :param ax: the axes to plot on; will be created if not specified
        :param name: which classifier to make plots for; if not specified the classifier with best evaluation metrics \
        is chosen
        """
        if name is None:
            name = self.best_model_name
        if ax is None:
            fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(self.best_clfs[name], self.X_test,
                                              self.y_test, ax=ax)
