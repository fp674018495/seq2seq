from keras.layers import LSTM,Activation,Dense,Input,GRU,RepeatVector,TimeDistributed
import numpy as np
BiLSTM=LSTM
from keras.models import Sequential,save_model,Model
import pandas as pd
from keras.layers.normalization import BatchNormalization
biLSTM=LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
# 基本参数
batch_size = 64
epochs = 50  #迭代次数
hide_num = 50  #隐层神经元个数
stride=8
out_stride=4

import seaborn as sns

# 数据集路径
data_path = 'data_k.csv'
# data_path = 'data_k30.csv'
#加载数据
with open(data_path) as file:
    data=pd.read_csv(file)
    data1=data.values[:,1]

#归一化


X=np.empty((len(data1)-stride,stride))
Y=np.empty(len(data1)-stride)
for i  in range(len(data1)-stride-1):
    Y[i]=data1[i+stride]
    X[i]=data1[i:i+stride]
range_y=np.max(Y)-np.min(Y)
#归一化
scaler=MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
Y=Y/(np.max(Y)-np.min(Y))

#扩维
X=np.expand_dims(X,2)




Y1=np.empty((stride,len(data1)-stride))
for i in range(stride):
    Y1[i]=Y
Y1=np.expand_dims(Y1.T,2)

#测试集训练集划分
num_split=int(len(Y)*0.8)
train_X=X[:num_split]
train_Y=Y1[:num_split]
test_X=X[num_split:]
test_Y=Y1[num_split:]

def show(test_y,pre_y):
    plt.plot(test_y*range_y)
    plt.plot(pre_y*range_y)
    # print(train_history.history)
    plt.title("BiLSTM2LSTM")
    plt.ylabel("客流量/人", fontproperties="SimSun")
    # plt.xlabel("15分钟/次", fontproperties="SimSun")
    plt.xlabel("30分钟/次", fontproperties="SimSun")
    plt.legend(["test","predict"],loc='upper left')
    plt.show()


def show_train_history(trian_history):
    plt.plot(trian_history.history["loss"])
    plt.plot(trian_history.history["val_loss"])
    # print(train_history.history)
    plt.title("BiLSTM2LSTM")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["loss","val_loss"],loc='upper left')
    plt.show()


def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return np.sum(s,axis=0)/len(s)

def show_heatmap(uniform_data):
    import seaborn as sns
    sns.set()
    np.random.seed(0)
    # 取出这三个属性画热力图，坐标点的位置是passengers
    y_tick = ["t-1","t-2","t-3","t-4"]
    x_tick =["t+1","t+2","t+3","t+4"]
    data = {}
    for i in range(len(uniform_data)):
        data[x_tick[i]] = uniform_data[i]
    pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    print(pd_data)
    ax = sns.heatmap(pd_data)
    plt.ylabel("InputSequence")
    plt.xlabel("OutputSequence")
    plt.show()
    pass


# 定义编码器的输入
# encoder_inputs (None, num_encoder_tokens), None表示可以处理任意长度的序列
encoder_inputs = Input(shape=(stride,1))

# 编码器，要求其返回状态
encoder = BiLSTM(hide_num, return_state=True)

# 调用编码器，得到编码器的输出（输入其实不需要），以及状态信息 state_c
encoder_outpus , state_h,state_c = encoder(encoder_inputs)

# 丢弃encoder_outputs, 我们只需要编码器的状态
encoder_state =[state_h, state_c]

 #定义解码器的输入
# 同样的，None表示可以处理任意长度的序列
decoder_inputs = Input(shape=(stride,1))

# 接下来建立解码器，解码器将返回整个输出序列
# 并且返回其中间状态，中间状态在训练阶段不会用到，但是在推理阶段将是有用的
decoder_lstm = LSTM(hide_num, return_sequences=True, return_state=True)

# 将编码器输出的状态作为初始解码器的初始状态
decoder_outputs, _,_ = decoder_lstm(decoder_inputs, initial_state=encoder_state)

# 添加全连接层
decoder_dense = Dense(units=1, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

# 定义整个模型
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
print(model.summary())
# 定义回调函数
#callback_list = [callbacks.EarlyStopping(patience=10)]
# 编译模型
model.compile(optimizer='adam', loss='mse')


train_history=model.fit([train_X,train_X], train_Y,batch_size=batch_size, epochs = epochs,validation_split=0.2)


pre_Y=model.predict([test_X,test_X])
pre_train=model.predict([train_X,train_X])


####################################################################
pre_Y1=np.empty((len(pre_train)-out_stride,out_stride))
a=pre_train[:,0][0:out_stride]
for i  in range(len(pre_train)-out_stride-1):
  #  print(pre_Y[:,0][i:i+20])
    pre_Y1[i]=pre_train[:,0][i:i+out_stride].reshape(out_stride)
s1 = softmax(pre_Y1.reshape(len(pre_Y1),-1))
s2 = softmax(train_X.reshape(len(train_X),-1))
print("s1",s1.shape)
print("s2",s2.shape)
uniform_data=np.multiply(s1.reshape(len(s1),1).T,s2[:4].reshape(len(s1),1))
max=np.max(uniform_data)
min=np.min(uniform_data)
show_heatmap(uniform_data)
######################################################################

show(test_Y[:,0],pre_Y[:,0])
show_train_history(train_history)
rmse=mean_squared_error(test_Y[:,0],pre_Y[:,0])*mean_squared_error(test_Y[:,0],pre_Y[:,0])
mae=mean_absolute_error(test_Y[:,0],pre_Y[:,0])
print("rmse:",rmse)
print("mae:",mae)

# 保存模型
model.save('s2s_2.h5')
