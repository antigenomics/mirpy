from sklearn.tree import DecisionTreeClassifier

GRID_SEARCH_PARAMS = {'svm': {'C': (1, 5, 10, 50, 100),
                              'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                              'probability': [True],
                              },

                      'ab': {'n_estimators': (10, 25, 50, 100, 125, 150, 200),
                             'estimator': (DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)),
                             },

                      'knn': {'n_neighbors': (3, 5, 10, 50, 75, 100),
                              'leaf_size': (1, 2, 3, 5, 10, 15, 20),
                              'weights': ['uniform', 'distance']},

                      'rfc': {'max_depth': (1, 2, 3),
                              'n_estimators': (50, 75, 100, 125, 150, 200),
                              'min_samples_leaf': (8, 10),
                              'oob_score': (False, True),
                              'n_jobs': [-1]},

                      'mlpclassifier': {'hidden_layer_sizes': (
                          (3000, 1500, 100, 60, 30, 10),
                          (3000, 1500, 150, 100, 50, 25, 10),
                          (2500, 2000, 150, 100, 50, 25, 10),
                          (3000, 1500, 100, 50, 25, 10)),
                          'alpha': (0.0001, 0.001, 0.01),
                          'learning_rate': ['adaptive'],
                          'max_iter': [1000]
                      },

                      'xgboost': {'n_estimators': (10, 25, 50, 75, 100),
                                  'subsample': (0.25, 0.5, 0.75, 1),
                                  'n_jobs': [-1]},
                      }
