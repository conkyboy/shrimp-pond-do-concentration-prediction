import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import h5py
from sklearn import metrics

# 构建模型
def create_model():
    model_net = tf.keras.Sequential()
    model_net.add(tf.keras.layers.Dense(445, input_shape=(12,), activation='relu', use_bias=True))  # 输入层，中间层
    # model_net.add(tf.keras.layers.Dense(100, activation='relu', use_bias=True))
    model_net.add(tf.keras.layers.Dense(1, use_bias=True))  # 输出层
    model_net.compile(loss='mse', optimizer='adam')  # 损失函数：均方差；优化器：adam
    return model_net

if __name__ == '__main__':
    # data = pd.read_csv(r'E:\A研一\毕业设计\数据\2020-2021meng.csv', header=None)
    data = pd.read_csv(r'E:\毕业设计\数据\加风向数据\总加风向进入约简20211013-20211125填充.csv', header=None)
    # Train = pd.read_csv(r'E:\A研一\毕业设计\数据\处理过的数据\Train\约简数据20211013-20211125.csv', header=None)
    # Test = pd.read_csv(r'E:\A研一\毕业设计\数据\处理过的数据\Test\约简数据20211013-20211125.csv', header=None)
    # data1 = MinMaxScaler().fit_transform(data)
    # data2 = pd.DataFrame(data1)
    x = data.iloc[:, :-1]
    print(x)
    y = data.iloc[:, -1]
    print(y)
    # x1 = MinMaxScaler().fit_transform(x)
    x1 = StandardScaler().fit_transform(x)
    print(x1)

    X_train, X_test, y_train, y_test = train_test_split(x1, y, test_size=0.1, random_state=532)#532
    model = create_model()
    model.fit(X_train, y_train, epochs=500, batch_size=23)  # 5000
    y_pre = model.predict(X_test)

    print('真实值:', y_test)
    print('预测值:', y_pre)
    print('11', type(y_test))
    print('12', type(y_pre))
    # 输出到csv
    # newDataframe_test = y_test.reset_index()
    # newDataframe_test.to_csv(r'E:\A研一\毕业设计\数据\作图数据\10分钟风速\test1.csv')
    # newDataframe_pre = pd.DataFrame(y_pre)
    # newDataframe_pre.to_csv(r'E:\A研一\毕业设计\数据\作图数据\10分钟风速\pre1.csv')
    # print('预测', p)

    MSE = metrics.mean_squared_error(y_test, y_pre)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pre))
    MAE = metrics.mean_absolute_error(y_test, y_pre)
    R2 = metrics.r2_score(y_test, y_pre)

    print('MSE:', MSE)
    print('RMSE:', RMSE)
    print('MAE:', MAE)
    print('R2:', R2)

    # 绘制折线图
    # plt.figure(figsize=(8, 10))
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    # plt.xlim(1, 12)
    # plt.ylim(0, 500)
    plt.plot(range(len(y_pre)), y_pre, label='预测值')
    plt.plot(range(len(y_test)), y_test, label='真实值')  # D代表菱形
    # 设置x轴的刻度, 刻度为0-70，步长为4，对应相应的values值;rotation：对应刻度转换，转为斜体
    plt.xticks(fontproperties='Times New Roman', size=15)
    # plt.figure(figsize=[20, 6])
    plt.yticks(fontproperties='Times New Roman', size=15)

    plt.xlabel('时间 Time(h)', size=15)  # 显示横坐标的标签
    plt.ylabel('温度 Temperature(℃)', size=15)  # 显示纵坐标的标签
    # plt.title('2020年钦州市月平均降水量')
    # 设置图例
    plt.legend(['预测值 Predictive value', '实测值 Measured value'], prop={'size': 13})
    # plt.savefig(r'E:\A研一\毕业设计\过程管理\中英文\约简有小时5.png', dpi=300)
    plt.show()

    # # Train = pd.read_csv(r'C:\Users\Administrator\Desktop\1028-1109\train01.csv', encoding='gbk')
    # # Test = pd.read_csv(r'C:\Users\Administrator\Desktop\1028-1109\test01.csv', encoding='gbk')
    # Train = pd.read_csv(r'C:\Users\Administrator\Desktop\水温预测\模型训练\预测前5天\最高+最低+平均气温+气压\train2021.11.2-2021.11.25.csv', encoding='gbk')
    # Test = pd.read_csv(r'C:\Users\Administrator\Desktop\水温预测\模型训练\预测前5天\最高+最低+平均气温+气压\test2021.10.28-2021.11.1.csv', encoding='gbk')
    # print('训练集', Train)
    # print('测试集', Test)
    # x = Train.iloc[:, 0:4]
    # # x = Train.iloc[:, 8]
    # # x = StandardScaler().fit_transform(x)
    # print('训练值：', x)
    # y = Train.iloc[:, -1]
    # # y = StandardScaler().fit_transform(y)
    # print('目标值：', y)
    # model = create_model()
    # m = model.fit(x, y, epochs=5000, batch_size=24)
    # print(m)
    # # test = Test.iloc[:, 8]
    # test = Test.iloc[:, 0:4]
    # # test = Test.iloc[:, 9]
    # print('test', test)
    # pre = model.predict(test)
    # print('测试：', pre)
    # # true1 = data.iloc[:10, 2]
    # # true2 = np.array(true1)
    # # print('真实值', true2)
    # true = Test.iloc[:, -1]
    # true = np.array(true)
    # print('true', true)
    # err = np.abs(pre - true).mean()
    # print('err', err)
    #
    # # 绘制折线图
    # # plt.figure(figsize=(8, 8))
    # plt.rcParams['font.sans-serif'] = 'SimHei'
    # plt.rcParams['axes.unicode_minus'] = False
    # # plt.xlim(1, 12)
    # # plt.ylim(0, 500)
    # plt.plot(pre)
    # plt.plot(true)  # D代表菱形
    # # 设置x轴的刻度, 刻度为0-70，步长为4，对应相应的values值;rotation：对应刻度转换，转为斜体
    # # plt.xticks(range(0, 120, 8))
    # # plt.figure(figsize=[20, 6])
    # plt.yticks(range(15, 28, 2))
    #
    # plt.xlabel('小时')  # 显示横坐标的标签
    # plt.ylabel('温度')  # 显示纵坐标的标签
    # # plt.title('2020年钦州市月平均降水量')
    # # 设置图例
    # plt.legend(['预测水温', '实测水温'])
    # # plt.savefig('D:\\PythonProject\\PythonBasis\\DataQ\\temp/气温折线图.png')
    # plt.show()
    # # err = np.abs(test1 - true).mean()
    # # print('err', err)


    # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=0)
    # model = create_model()  # 调用模型
    # print(model.summary())  # 输出模型信息
    # model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100) # 训练模型


    # 获取数据和标签
    # train_x, train_y, test_x, test_y = load_data('D:\PythonProject\PythonBasis\DataTWD\iris.data')
    # 调用基于全连接网络的序贯模型
    # model = create_model()
    # 输出模型信息
    # print(model.summary())
    # 训练模型
    # model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=32, verbose=2)
    """
      validation_data:验证集
      epochs：训练轮数
      batch_size:每多少个样本为一批进行小批量进行梯度下降
      verbose：日志显示，
    """

