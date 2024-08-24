import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler  # 标准差
from sklearn.preprocessing import MinMaxScaler  # 极差标准化、归一化
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB, CategoricalNB

# 数据归一化
def Data_standard(Data):
    newDataFrame = pd.DataFrame(index=Data.index)
    columns = Data.columns.tolist()
    for c in columns:
        d = Data[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    newDataFrame = round(newDataFrame, 4)
    return newDataFrame

# 多邻域半径集/标准差
def Radii(radii_data):
    radii = round(radii_data.iloc[:, :-1].std(), 4)
    return radii

# 计算多邻域距离矩阵（修改）
def Distance_matrix(Data, j):
    '''
    :param Data_standard: 归一化后的数据集
    :param j: 任意一个特征
    :return: 邻域矩阵
    '''
    Data1 = np.array(Data)
    m = Data1[:, j]
    raddi = round(Data.loc[:, j].std(), 4)
    w = m.reshape(m.shape[0], 1)
    distance = pdist(w, metric='euclidean')
    matrix = squareform(distance)
    eighborhood_matrex = np.where(matrix <= raddi, 1, 0)
    return eighborhood_matrex
# 计算多邻域距离矩阵
# def Distance_matrix(Data, j):
#     '''
#     :param Data_standard: 归一化后的数据集
#     :param j: 任意一个特征
#     :return: 邻域矩阵
#     '''
#     NR = np.full((Data.shape[0], Data.shape[0]), 0)
#     raddi = round(Data.loc[:, j].std(), 4)
#     List = list(Data.loc[:, j])
#     for i in range(len(List)):
#         for k in range(len(List)):
#             if np.linalg.norm(List[i]-List[k]) <= raddi: #0.3446
#                 NR[i][k] = 1
#             else:
#                 NR[i][k] = 0
#     return NR
    #return raddi
# 邻域熵
def Neighborhood_entropy(M):
    '''
    :param M: M是一个矩阵
    :return:返回邻域熵，结果是一个数
    '''
    M = pd.DataFrame(M)
    U = len(M)
    # row = M.index
    col = M.columns
    sum = 0
    for j in col:
        sum = sum + np.log2(np.sum(M.loc[:, j])/U)
    return -sum/U


# 邻域联合熵(修改版)
def Neighborhood_joint_entropy(M1, Md):
    M1 = pd.DataFrame(M1)
    Md = pd.DataFrame(Md)
    M_new = np.bitwise_and(M1, Md)

    U = len(M_new)
    row = M_new.index
    col = M_new.columns
    sum = 0
    for j in col:
        sum = sum + np.log2(np.sum(M_new.loc[:, j]) / U)
    return -sum/U
# 邻域联合熵
# def Neighborhood_joint_entropy(M1, Md):
#     M1 = pd.DataFrame(M1)
#     Md = pd.DataFrame(Md)
#     M_new = pd.DataFrame((M1.shape[0], M1.shape[1]))
#     for i in M1.index:
#         for j in M1.columns:
#             M_new.loc[i, j] = M1.loc[i, j] & Md.loc[i, j]
#
#     U = len(M_new)
#     row = M_new.index
#     col = M_new.columns
#     sum = 0
#     for j in col:
#         sum = sum + np.log2(np.sum(M_new.loc[:, j]) / U)
#     return -sum/U

# # 邻域联合熵(三个参数)
# def Neighborhood_joint_entropy2(M1, M2, Md):
#     M1 = pd.DataFrame(M1)
#     M2 = pd.DataFrame(M2)
#     Md = pd.DataFrame(Md)
#     M_new = pd.DataFrame((M1.shape[0], M1.shape[1]))
#     for i in M1.index:
#         for j in M1.columns:
#             M_new.loc[i, j] = M1.loc[i, j] & M2.loc[i, j] & Md.loc[i, j]
#
#     U = len(M_new)
#     row = M_new.index
#     col = M_new.columns
#     sum = 0
#     for j in col:
#         sum = sum + np.log2(np.sum(M_new.loc[:, j]) / U)
# #   return -sum/U

# 邻域互信息
def Neighborhood_mutual_information(M1, Md):
    return Neighborhood_entropy(M1) + Neighborhood_entropy(Md) - Neighborhood_joint_entropy(M1, Md)


# 邻域条件互信息（修改版）
def Neighborhood_conditional_mutual_information(M1, M2, Md):
    """
    NCMI(M2;Md|M1)=NE(M2,M1)+NE(M1,Md)-NE(M1,M2,Md)-NE(M1)
    """
    M1 = pd.DataFrame(M1)
    M2 = pd.DataFrame(M2)
    Md = pd.DataFrame(Md)
    M_new = np.bitwise_and(np.bitwise_and(M1, M2), Md)

    U = len(M_new)
    row = M_new.index
    col = M_new.columns
    sum = 0
    for j in col:
        sum = sum + np.log2(np.sum(M_new.loc[:, j]) / U)
    temp = -sum / U
    return Neighborhood_joint_entropy(M2, M1) + Neighborhood_joint_entropy(M1, Md) - temp -Neighborhood_entropy(M1)
# 邻域条件互信息
# def Neighborhood_conditional_mutual_information(M1, M2, Md):
#     """
#     NCMI(M2;Md|M1)=NE(M2,M1)+NE(M1,Md)-NE(M1,M2,Md)-NE(M1)
#     """
#     M1 = pd.DataFrame(M1)
#     M2 = pd.DataFrame(M2)
#     Md = pd.DataFrame(Md)
#     M_new = pd.DataFrame((M1.shape[0], M1.shape[1]))
#     for i in M1.index:
#         for j in M1.columns:
#             M_new.loc[i, j] = M1.loc[i, j] & M2.loc[i, j] & Md.loc[i, j]
#     U = len(M_new)
#     row = M_new.index
#     col = M_new.columns
#     sum = 0
#     for j in col:
#         sum = sum + np.log2(np.sum(M_new.loc[:, j]) / U)
#     temp = -sum / U
#     return Neighborhood_joint_entropy(M2, M1) + Neighborhood_joint_entropy(M1, Md) - temp -Neighborhood_entropy(M1)

# 最大依赖最小冗余最大交互函数
def SIG(F_j_F, Red):
    sum1 = Neighborhood_mutual_information(Distance_matrix(Data_set, F_j_F), Distance_matrix(Data_set, Data_set.shape[1] - 1))
    sum2 = 0
    for fs in Red:
        sum2 = sum2 + Neighborhood_mutual_information(Distance_matrix(Data_set, F_j_F), Distance_matrix(Data_set, fs))
    sum2 = sum2 * (1/len(Red))
    sum3 = 0
    for F_j_F_ in F-{F_j_F}:
        sum3 = sum3 + Neighborhood_conditional_mutual_information(Distance_matrix(Data_set, F_j_F), Distance_matrix(Data_set, F_j_F_), Distance_matrix(Data_set, Data_set.shape[1]-1))
    sum3 = sum3 * (1/len(F))
    return sum1 - sum2 + sum3


if __name__ == '__main__':
    df = pd.read_csv(r'D:\Datasets\实验分类数据集\Balance-scale.csv', header=None)
    # Data_set = pd.read_csv(r'E:\溶解氧预测\溶解氧数据\总无小时20211013-20211125.csv', header=None)
    #众数填充
    # imp = SimpleImputer(missing_values='?', strategy="most_frequent")
    # imp.fit(df)
    # data = imp.transform(df)
    # df = pd.DataFrame(data)

    print(df)
    # df1 = MinMaxScaler().fit_transform(df.iloc[:, :-1])
    df1 = df.iloc[:, :-1]
    print('df1:', df1)
    Data_set = pd.merge(df1, df.iloc[:, -1], left_index=True, right_index=True)
    print('Data_set:', Data_set)

    # print(df)
    # df1 = MinMaxScaler().fit_transform(df)
    # Data_set = pd.DataFrame(df1)
    # print('Data_set:', Data_set)
    # Data_set = pd.merge(df1, df.iloc[:, -1], left_index=True, right_index=True)
    # print('Data_set:', Data_set)


    F = set(Data_set.columns[:-1])
    List_relevance = []  # 计算每个特征的依赖值
    Red = []  # 已选特征 最大依赖值对应的特征
    for i in range(Data_set.shape[1] - 1):
        Relevance = Neighborhood_mutual_information(Distance_matrix(Data_set, i), Distance_matrix(Data_set, Data_set.shape[1]-1))
        List_relevance.append(Relevance)
    fs = List_relevance.index(max(List_relevance))
    Red.append(fs)
    print('第一次约简：', Red)
    F.remove(fs)
    print('剩余待选特征：', F)
    #print(Data_set.iloc[:, fs])
    while F:
        Dic = dict()  # 构建空字典
        for F_j_F in F:
            print(F_j_F)
            Dic[F_j_F] = SIG(F_j_F, set(Red))  # 获取字典的值
        fs_next = max(Dic, key=Dic.get)
        Red.append(fs_next)
        F.remove(fs_next)
        print('约简：', Red, len(Red))
        X1 = Data_set.iloc[:, Red]
        # print('X1:', X1)
        # y1 = Data_set.iloc[:, -1]
        Y1 = Data_set.iloc[:, -1]
        # print('y1:', y1)

        # Y1 = y1.astype(int)
        # Y1 = y1.astype(int)
        print('Y1:', Y1)
        """SVM"""
        SVM = svm.SVC(C=1, kernel='rbf', gamma='auto', cache_size=5000)  # svc用于做分类任务
        # clf1 = svm.SVR(C=1, kernel='linear', gamma='auto', cache_size=5000)  # svr用于做回归任务
        scores1 = cross_val_score(SVM, X1, Y1, cv=10)  # 10折交叉验证
        # print('精度：', scores)
        mean1 = round(np.mean(scores1), 4)
        std1 = round(np.std(scores1), 2)
        print('SVM:', mean1, std1)
        # """KNN"""
        # KNN = KNeighborsClassifier(n_neighbors=3)  # knn分类器
        # # knn = KNeighborsRegressor(n_neighbors=3)  # knn回归
        # # clf2 = knn.fit(X1, y1)
        # scores2 = cross_val_score(KNN, X1, Y1, cv=10)
        # mean2 = round(np.mean(scores2), 4)
        # std2 = round(np.std(scores2), 2)
        # print('KNN:', mean2, std2)
        '''CART'''
        CART = tree.DecisionTreeClassifier(criterion="entropy", random_state=30, splitter="random")  # 实例化模型 分类任务
        # clf3 = tree.DecisionTreeRegressor()
        # clf3 = clf1.fit(X1, y1)  # 训练模型，得到模型clf1
        # score = clf1.score(Xtest, Ytest)  # 返回预测的准确度
        score3 = cross_val_score(CART, X1, Y1, cv=10)
        mean3 = round(np.mean(score3), 4)
        std3 = round(np.std(score3), 2)
        print('CART:', mean3, std3)
        """NB"""
        NB = GaussianNB()
        score4 = cross_val_score(NB, X1, Y1, cv=10)
        mean4 = round(np.mean(score4), 4)
        std4 = round(np.std(score4), 2)
        print('NB:',  mean4, std4)
        """average"""
        aver = round((mean1+mean3+mean4)/3, 4)
        std = round((std1+std3+std4)/3, 4)
        print('平均精度aver:', aver, std)

        # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X1, Y1, test_size=0.3, random_state=35)
        # clf = svm.SVC(C=1.0, kernel="rbf", gamma="auto", cache_size=200).fit(Xtrain, Ytrain)
        # print("SVM分类器的训练分类精度：", clf.score(Xtrain, Ytrain))
        # print("SVM分类器的测试分类精度：", clf.score(Xtest, Ytest))
        # # 取分类精度最大的约简数量



# print(Red)
# print(F)




    # A = Set.index(np.max(Set))
    # print(A)
    # print(A)

        # """计算待选特征与决策属性之间的相关性"""
        # # print('相关性：', F_j_F)
        # sum1 = Neighborhood_mutual_information(Distance_matrix(Data_set, F_j_F), Distance_matrix(Data_set, Data_set.shape[1] - 1))
        # # list1.append(relevance)
        # # print(list1)
        # """计算待选特征与已选特征之间的冗余性"""
        # sum2 = 0
        # for fs in Red:
        #     # print('冗余性：', fs)
        #     sum2 = sum2 + Neighborhood_mutual_information(Distance_matrix(Data_set, F_j_F), Distance_matrix(Data_set, fs))
        #     #list2.append(Redundancy)
        #     # print(list2)
        # sum2 = sum2 * (1 / len(Red))
        # sum3 = 0
        # for F_j_F_ in F-{F_j_F}:
        #     """计算剩余待选特征与待选特征以及决策属性之间的交互性"""
        #     # print('交互性：', F_j_F_)
        #     sum3 = sum3 + Neighborhood_conditional_mutual_information(Distance_matrix(Data_set, F_j_F), Distance_matrix(Data_set, F_j_F_), Distance_matrix(Data_set, Data_set.shape[1]-1))
        #     # list3.append(Itr)
        #     # print(list3)
        # sum3 = sum3 * (1/len(F))
        # print(sum1 - sum2 + sum3)










