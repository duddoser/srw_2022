import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pandas.read_csv("vgsales.csv")
# для построения лог. регрессии были выбраны столбцы 'Genre', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'
data_sel = data.loc[:, data.columns.isin(['Platform', 'Genre', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])]
data_sel = data_sel.dropna()
# строим классификацию по столбцу Platform, PS2 - класс 0, остальные платформы - класс 1
data_sel['Platform'] = np.where(data_sel['Platform'] == 'PS2', 0, 1)

# обработаем столбец Genre и присвоим различным значениям столбца цифру от 0 до 6
data_sel['Genre'] = np.where(data_sel['Genre'] == 'Action', 0, data_sel['Genre'])
data_sel['Genre'] = np.where(data_sel['Genre'] == 'Adventure', 1, data_sel['Genre'])
data_sel['Genre'] = np.where(data_sel['Genre'] == 'Sports', 2, data_sel['Genre'])
data_sel['Genre'] = np.where(data_sel['Genre'] == 'Platform', 3, data_sel['Genre'])
data_sel['Genre'] = np.where(data_sel['Genre'] == 'Racing', 4, data_sel['Genre'])
data_sel['Genre'] = np.where(data_sel['Genre'] == 'Fighting', 5, 6)

# убираем из данных столбец Platform, по которому строим классификацию
platforms = data_sel.loc[:, data_sel.columns.isin(['Platform'])]
x = data_sel.loc[:, data_sel.columns.isin(['Genre', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])]
# разделим набор данных на тестовую и обучающую выборку
x_train, x_test, y_train, y_test = train_test_split(x, platforms, test_size=0.3)

# строим лог. регрессию
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# print(clf.score(x_test, y_test))
print("accuracy: " + str(np.average(cross_val_score(clf, x_test, y_test, scoring='accuracy'))))
# accuracy: 0.8687488274165641
print("f1: " + str(np.average(cross_val_score(clf, x_test, y_test, scoring='f1'))))
# f1: 0.9297651473118999
print("precision: " + str(np.average(cross_val_score(clf, x_test, y_test, scoring='precision'))))
# precision: 0.8687488274165641
print("recall: " + str(np.average(cross_val_score(clf, x_test, y_test, scoring='recall'))))
# recall : 1.0

grid_search_cv = GridSearchCV(cv=3, error_score='raise',
       estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=42,
            splitter='best'), n_jobs=None,
       param_grid={'max_depth': list(range(2, 20)), 'min_samples_split': [2, 3, 4]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=1)
grid_search_cv.fit(x_train, y_train)

param_grid = {'n_estimators': [200, 300, 400], 'max_features': ['auto'],
               'max_depth': list(range(1, 20)), 'criterion': ['gini']}

# построим классификатор типа Случайный Лес
RFC = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, refit=True)
RFC.fit(x_train, y_train)

print("accuracy: " + str(np.average(cross_val_score(grid_search_cv.best_estimator_, x_test, y_test, scoring='accuracy'))))
# accuracy: 0.9328435930041067
print("f1: " + str(np.average(cross_val_score(grid_search_cv.best_estimator_, x_test, y_test, scoring='f1'))))
# f1 : 0.9615368345963002
print("precision: " + str(np.average(cross_val_score(grid_search_cv.best_estimator_, x_test, y_test, scoring='precision'))))
# precision: 0.9603767058423113
print("recall: " + str(np.average(cross_val_score(grid_search_cv.best_estimator_, x_test, y_test, scoring='recall'))))
# recall: 0.9627700398537327

# сравнивая значения метрик для классификаторов типа лог. регрессии и случайного леса,
# делаем вывод, что второй построенный классификатор лучше
