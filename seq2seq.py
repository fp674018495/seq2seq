from keras.layers import LSTM,Activation,Dense,Input,GRU,RepeatVector,TimeDistributed
import numpy as np
BiGRU=GRU
from keras.models import Sequential,save_model,Model
import pandas as pd
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
# 基本参数
batch_size = 64
epochs = 200  #迭代次数
hide_num = 50  #隐层神经元个数


# 数据集路径
# data_path = 'data_k.csv'
data_path = 'data_k30.csv'
#加载数据
with open(data_path) as file:
    data=pd.read_csv(file)
    data1=data.values[:,1]

#归一化
# scaler=np.max(data1)-np.min(data1)
# data1=data1/scaler

X=np.empty((len(data1)-20,20))
Y=np.empty(len(data1)-20)
for i  in range(len(data1)-21):
    Y[i]=data1[i+20]
    X[i]=data1[i:i+20]
range_y=np.max(Y)-np.min(Y)
#归一化
scaler=MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
Y=Y/(np.max(Y)-np.min(Y))
#扩维
X=np.expand_dims(X,2)

Y1=np.empty((20,len(data1)-20))
for i in range(20):
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
    plt.title("BiGRU2GRU")
    plt.ylabel("客流量/人",fontproperties="SimSun")
    # plt.xlabel("15分钟/次",fontproperties="SimSun")
    plt.xlabel("30分钟/次",fontproperties="SimSun")
    plt.legend(["test","predict"],loc='upper left')
    plt.show()


def show_train_history(trian_history):
    plt.plot(trian_history.history["loss"])
    plt.plot(trian_history.history["val_loss"])
    # print(train_history.history)
    plt.title("BiGRU2GRU")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["loss","val_loss"],loc='upper left')
    plt.show()

# 定义编码器的输入
# encoder_inputs (None, num_encoder_tokens), None表示可以处理任意长度的序列
encoder_inputs = Input(shape=(20,1))

# 编码器，要求其返回状态
encoder = BiGRU(hide_num, return_state=True)

# 调用编码器，得到编码器的输出（输入其实不需要），以及状态信息 state_c
encoder_outpus,state_c = encoder(encoder_inputs)

# 丢弃encoder_outputs, 我们只需要编码器的状态
encoder_state = state_c

 #定义解码器的输入
# 同样的，None表示可以处理任意长度的序列
decoder_inputs = Input(shape=(20,1))

# 接下来建立解码器，解码器将返回整个输出序列
# 并且返回其中间状态，中间状态在训练阶段不会用到，但是在推理阶段将是有用的
decoder_lstm = GRU(hide_num, return_sequences=True, return_state=True)

# 将编码器输出的状态作为初始解码器的初始状态
decoder_outputs, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)

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

show(test_Y[:,0],pre_Y[:,0])
show_train_history(train_history)
rmse=mean_squared_error(test_Y[:,0],pre_Y[:,0])*mean_squared_error(test_Y[:,0],pre_Y[:,0])
mae=mean_absolute_error(test_Y[:,0],pre_Y[:,0])
print("rmse:",rmse)
print("mae:",mae)

# 保存模型
model.save('s2s_1.h5')
