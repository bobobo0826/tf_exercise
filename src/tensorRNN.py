# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:49:52 2018

@author: zy
"""

'''
使用TensorFlow库实现单层RNN  分别使用LSTM单元，GRU单元，static_rnn和dynamic_rnn函数
'''

import tensorflow as tf
import numpy as np
tf.reset_default_graph()

'''
一 使用动态RNN处理变长序列
'''
np.random.seed(0)

#创建输入数据  正态分布 2：表示一次的批次数量 4：表示时间序列总数  5：表示具体的数据
X = np.random.randn(2,4,5)

#第二个样本长度为3
X[1,1:] = 0
#每一个输入序列的长度
seq_lengths = [4,1]
print('X:\n',X)

#分别建立一个LSTM与GRU的cell，比较输出的状态  3是隐藏层节点的个数
cell = tf.contrib.rnn.BasicLSTMCell(num_units = 3,state_is_tuple = True)
gru = tf.contrib.rnn.GRUCell(3)

#如果没有initial_state，必须指定a dtype
outputs,last_states = tf.nn.dynamic_rnn(cell,X,seq_lengths,dtype =tf.float64 )
gruoutputs,grulast_states = tf.nn.dynamic_rnn(gru,X,seq_lengths,dtype =tf.float64 )

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

result,sta,gruout,grusta = sess.run([outputs,last_states,gruoutputs,grulast_states])

print('全序列:\n',result[0])
print('短序列:\n',result[1])

#由于在BasicLSTMCell设置了state_is_tuple是True，所以lstm的值为 (状态ct,输出h）
print('LSTM的状态:',len(sta),'\n',sta[1])  

print('GRU的全序列：\n',gruout[0])
print('GRU的短序列：\n',gruout[1])
#GRU没有状态输出，其状态就是最终输出，因为批次是两个，所以输出为2
print('GRU的状态:',len(grusta),'\n',grusta[1]) 




'''
二 构建单层单向RNN网络对MNIST数据集分类
'''
'''
MNIST数据集一个样本长度为28 x 28 
我们可以把一个样本分成28个时间段，每段内容是28个值，然后送入LSTM或者GRU网络
我们设置隐藏层的节点数为128
'''


def single_layer_static_lstm(input_x,n_steps,n_hidden):
    '''
    返回静态单层LSTM单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    
    #把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....] 
    #如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度 
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)

    #可以看做隐藏层
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    #静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量 
    hiddens,states = tf.contrib.rnn.static_rnn(cell=lstm_cell, inputs=input_x1, dtype=tf.float32)

    return hiddens,states


def single_layer_static_gru(input_x,n_steps,n_hidden):
    '''
    返回静态单层GRU单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    
    #把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....] 
    #如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度 
    input_x1 = tf.unstack(input_x,num=n_steps,axis=1)

    #可以看做隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    #静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量 
    hiddens,states = tf.contrib.rnn.static_rnn(cell=gru_cell,inputs=input_x1,dtype=tf.float32)
        
    return hiddens,states


def single_layer_dynamic_lstm(input_x,n_steps,n_hidden):
    '''
    返回动态单层LSTM单元的输出，以及cell状态
    
    args:
        input_x:输入张量  形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    #可以看做隐藏层
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    #动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens,states = tf.nn.dynamic_rnn(cell=lstm_cell,inputs=input_x,dtype=tf.float32)

    #注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens,[1,0,2])
    return hiddens,states



def single_layer_dynamic_gru(input_x,n_steps,n_hidden):
    '''
    返回动态单层GRU单元的输出，以及cell状态
    
    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    
    #可以看做隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    #动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens,states = tf.nn.dynamic_rnn(cell=gru_cell,inputs=input_x,dtype=tf.float32)
        
    
    #注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens,[1,0,2])    
    return hiddens,states


def  mnist_rnn_classfication(flag):
    '''
    对MNIST进行分类
    
    arg:
        flags:表示构建的RNN结构是哪种
            1：单层静态LSTM
            2: 单层静态GRU
            3：单层动态LSTM
            4: 单层动态GRU
    '''




    '''
    1. 导入数据集
    '''
    tf.reset_default_graph()
    from tensorflow.examples.tutorials.mnist import input_data
    
    #mnist是一个轻量级的类，它以numpy数组的形式存储着训练，校验，测试数据集  one_hot表示输出二值化后的10维
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    
    print(type(mnist)) #<class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>
    
    print('Training data shape:',mnist.train.images.shape)           #Training data shape: (55000, 784)
    print('Test data shape:',mnist.test.images.shape)                #Test data shape: (10000, 784)
    print('Validation data shape:',mnist.validation.images.shape)    #Validation data shape: (5000, 784)
    print('Training label shape:',mnist.train.labels.shape)          #Training label shape: (55000, 10)
    
    '''
    定义参数，以及网络结构
    '''
    n_input = 28             #LSTM单元输入节点的个数
    n_steps = 28             #序列长度
    n_hidden = 128           #LSTM单元输出节点个数(即隐藏层个数)
    n_classes = 10           #类别
    batch_size = 128         #小批量大小
    training_step = 5000     #迭代次数
    display_step  = 200      #显示步数
    learning_rate = 1e-4     #学习率  
    
    
    #定义占位符
    #batch_size：表示一次的批次样本数量batch_size  n_steps：表示时间序列总数  n_input：表示一个时序具体的数据长度  即一共28个时序，一个时序送入28个数据进入LSTM网络
    input_x = tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_input])
    input_y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])


    #可以看做隐藏层
    if  flag == 1:
        print('单层静态LSTM网络：')
        hiddens,states = single_layer_static_lstm(input_x,n_steps,n_hidden)
    elif flag == 2:
        print('单层静态gru网络：')
        hiddens,states = single_layer_static_gru(input_x,n_steps,n_hidden)
    elif  flag == 3:
        print('单层动态LSTM网络：')
        hiddens,states = single_layer_dynamic_lstm(input_x,n_steps,n_hidden)
    elif flag == 4:
        print('单层动态gru网络：')
        hiddens,states = single_layer_dynamic_gru(input_x,n_steps,n_hidden)
                
    print('hidden:',hiddens[-1].shape)      #(128,128)
    
    #取LSTM最后一个时序的输出，然后经过全连接网络得到输出值
    output = tf.contrib.layers.fully_connected(inputs=hiddens[-1],num_outputs=n_classes,activation_fn = tf.nn.softmax)
    
    '''
    设置对数似然损失函数
    '''
    #代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    cost = tf.reduce_mean(-tf.reduce_sum(input_y*tf.log(output),axis=1))
    
    '''
    求解
    '''
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    #预测结果评估
    #tf.argmax(output,1)  按行统计最大值得索引
    correct = tf.equal(tf.argmax(output,1),tf.argmax(input_y,1))       #返回一个数组 表示统计预测正确或者错误 
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))             #求准确率
    
    
    #创建list 保存每一迭代的结果
    test_accuracy_list = []
    test_cost_list=[]
    
    
    with tf.Session() as sess:
        #使用会话执行图
        sess.run(tf.global_variables_initializer())   #初始化变量    
        
        #开始迭代 使用Adam优化的随机梯度下降法
        for i in range(training_step): 
            x_batch,y_batch = mnist.train.next_batch(batch_size = batch_size)   
            #Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1,n_steps,n_input])
            
            #开始训练
            train.run(feed_dict={input_x:x_batch,input_y:y_batch})   
            if (i+1) % display_step == 0:
                 #输出训练集准确率        
                training_accuracy,training_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch,input_y:y_batch})   
                print('Step {0}:Training set accuracy {1},cost {2}.'.format(i+1,training_accuracy,training_cost))
        
        
        #全部训练完成做测试  分成200次，一次测试50个样本
        #输出测试机准确率   如果一次性全部做测试，内容不够用会出现OOM错误。所以测试时选取比较小的mini_batch来测试
        for i in range(200):        
            x_batch,y_batch = mnist.test.next_batch(batch_size = 50)      
            #Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1,n_steps,n_input])
            test_accuracy,test_cost = sess.run([accuracy,cost],feed_dict={input_x:x_batch,input_y:y_batch})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost) 
            if (i+1)% 20 == 0:
                 print('Step {0}:Test set accuracy {1},cost {2}.'.format(i+1,test_accuracy,test_cost)) 
        print('Test accuracy:',np.mean(test_accuracy_list))


if __name__ == '__main__':
    mnist_rnn_classfication(1)    #1：单层静态LSTM
    mnist_rnn_classfication(2)    #2：单层静态gru
    mnist_rnn_classfication(3)    #3：单层动态LSTM
    mnist_rnn_classfication(4)    #4：单层动态gru