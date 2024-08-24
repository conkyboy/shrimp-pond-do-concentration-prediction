# 导入库
import numpy as np  # numpy库
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.model_selection import train_test_split
#from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
from sklearn import metrics
import matplotlib.pyplot as plt  # 导入图形展示库
# 数据准备


def Model_BayesianRidge(X_train, X_test, y_train, y_test):
    model_br = BayesianRidge()
    model_br.fit(X_train, y_train)  # 5000
    y_pre = model_br.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\BR_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\BR_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)
    print("------------BayesianRidge--------------")

    print('BR_MSE:', MSE)
    print('BR_RMSE:', RMSE)
    print('BR_MAE:', MAE)
    print('BR_R2:', R2)

def Model_LinearRegression(X_train, X_test, y_train, y_test):
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)  # 5000
    y_pre = model_lr.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\LR_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\LR_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------LinearRegression--------------")

    print('LR_MSE:', MSE)
    print('LR_RMSE:', RMSE)
    print('LR_MAE:', MAE)
    print('LR_R2:', R2)


def Model_SVR(X_train, X_test, y_train, y_test):
    model_svr = LinearRegression()
    model_svr.fit(X_train, y_train)  # 5000
    y_pre = model_svr.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\SVR_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\SVR_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------SVR--------------")

    print('SVR_MSE:', MSE)
    print('SVR_RMSE:', RMSE)
    print('SVR_MAE:', MAE)
    print('SVR_R2:', R2)

def Model_KNeighborsRegressor(X_train, X_test, y_train, y_test):
    model_knn = KNeighborsRegressor()
    model_knn.fit(X_train, y_train)  # 5000
    y_pre = model_knn.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\KNN_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\KNN_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------KNN--------------")

    print('KNN_MSE:', MSE)
    print('KNN_RMSE:', RMSE)
    print('KNN_MAE:', MAE)
    print('KNN_R2:', R2)


def Model_Lasso(X_train, X_test, y_train, y_test):
    model_lasso = Lasso()
    model_lasso.fit(X_train, y_train)  # 5000
    y_pre = model_lasso.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\Lasso_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\Lasso_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------Lasso--------------")

    print('Lasso_MSE:', MSE)
    print('Lasso_RMSE:', RMSE)
    print('Lasso_MAE:', MAE)
    print('Lasso_R2:', R2)

def Model_ExtraTreeRegressor(X_train, X_test, y_train, y_test):
    model_extratree = ExtraTreeRegressor()
    model_extratree.fit(X_train, y_train)  # 5000
    y_pre = model_extratree.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\ExtraTree_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\ExtraTree_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------ExtraTree--------------")

    print('Lasso_MSE:', MSE)
    print('Lasso_RMSE:', RMSE)
    print('Lasso_MAE:', MAE)
    print('Lasso_R2:', R2)

def Model_BaggingRegressor(X_train, X_test, y_train, y_test):
    model_bagging = BaggingRegressor()
    model_bagging.fit(X_train, y_train)  # 5000
    y_pre = model_bagging.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\Bagging_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\Bagging_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------Bagging--------------")

    print('Bagging_MSE:', MSE)
    print('Bagging_RMSE:', RMSE)
    print('Bagging_MAE:', MAE)
    print('Bagging_R2:', R2)

def Model_RandomForestRegressor(X_train, X_test, y_train, y_test):
    model_randomforest = RandomForestRegressor()
    model_randomforest.fit(X_train, y_train)  # 5000
    y_pre = model_randomforest.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\RandomForest_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\RandomForest_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------RandomForest--------------")

    print('RandomForest_MSE:', MSE)
    print('RandomForest_RMSE:', RMSE)
    print('RandomForest_MAE:', MAE)
    print('RandomForest_R2:', R2)


def Model_RidgeRegressor(X_train, X_test, y_train, y_test):
    model_ridge = Ridge()
    model_ridge.fit(X_train, y_train)  # 5000
    y_pre = model_ridge.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\RidgeRegressor_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\RidgeRegressor_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------RidgeRegressor--------------")

    print('RidgeRegressor_MSE:', MSE)
    print('RidgeRegressor_RMSE:', RMSE)
    print('RidgeRegressor_MAE:', MAE)
    print('RidgeRegressor_R2:', R2)


def Model_DecisionTreeRegressor(X_train, X_test, y_train, y_test):
    model_decisiontree = DecisionTreeRegressor()
    model_decisiontree.fit(X_train, y_train)  # 5000
    y_pre = model_decisiontree.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\DecisionTree_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\DecisionTree_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------DecisionTree--------------")

    print('DecisionTree_MSE:', MSE)
    print('DecisionTree_RMSE:', RMSE)
    print('DecisionTree_MAE:', MAE)
    print('DecisionTree_R2:', R2)

def Model_GradientBoostingRegressor(X_train, X_test, y_train, y_test):
    model_gradientboosting = GradientBoostingRegressor()
    model_gradientboosting.fit(X_train, y_train)  # 5000
    y_pre = model_gradientboosting.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\GradientBoosting_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\GradientBoosting_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------GradientBoosting--------------")

    print('GradientBoosting_MSE:', MSE)
    print('GradientBoosting_RMSE:', RMSE)
    print('GradientBoosting_MAE:', MAE)
    print('GradientBoosting_R2:', R2)


def Model_AdaBoostRegressor(X_train, X_test, y_train, y_test):
    model_adaboostRegressor = AdaBoostRegressor()
    model_adaboostRegressor.fit(X_train, y_train)  # 5000
    y_pre = model_adaboostRegressor.predict(X_test)

    # print('真实值:', y_test)
    # print('预测值:', y_pre)
    # 输出到csv
    newDataframe_test = y_test.reset_index()
    newDataframe_test.to_csv(r'D:\ExperimentalData\ModelResult\AdaBoost_test.csv')
    newDataframe_pre = pd.DataFrame(y_pre)
    newDataframe_pre.to_csv(r'D:\ExperimentalData\ModelResult\AdaBoost_pre.csv')

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print("------------AdaBoost--------------")

    print('AdaBoost_MSE:', MSE)
    print('AdaBoost_RMSE:', RMSE)
    print('AdaBoost_MAE:', MAE)
    print('AdaBoost_R2:', R2)


if __name__ == '__main__':
    raw_data = pd.read_csv(r'D:\ExperimentalData\Experiment.csv', header=None)
    X = raw_data.iloc[:, :-1]  # 分割自变量
    y = raw_data.iloc[:, -1]  # 分割因变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=532)  # 532
    Model_BayesianRidge(X_train, X_test, y_train, y_test)
    Model_LinearRegression(X_train, X_test, y_train, y_test)
    Model_SVR(X_train, X_test, y_train, y_test)
    Model_KNeighborsRegressor(X_train, X_test, y_train, y_test)
    Model_Lasso(X_train, X_test, y_train, y_test)
    Model_ExtraTreeRegressor(X_train, X_test, y_train, y_test)
    Model_RandomForestRegressor(X_train, X_test, y_train, y_test)
    Model_RidgeRegressor(X_train, X_test, y_train, y_test)
    Model_DecisionTreeRegressor(X_train, X_test, y_train, y_test)
    Model_GradientBoostingRegressor(X_train, X_test, y_train, y_test)
    Model_BaggingRegressor(X_train, X_test, y_train, y_test)
    Model_AdaBoostRegressor(X_train, X_test, y_train, y_test)




