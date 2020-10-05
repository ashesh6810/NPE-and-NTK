from __future__ import division
import tensorflow as tf
import numpy as np
import math, random
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
N = 256
M=10000
a = 0
b = 1.0
ti=0
to=2
batch_size=100

X = np.arange(a, b, float((b-a)/N)).reshape((N,1))
x_old=np.tile(X,(M,1))

Y = np.zeros(N)
y_old=np.tile(Y,(M,1)).reshape(M*N,1)

T=np.arange(ti,to,float((to-ti)/M)).reshape((M,1))
t_old=np.zeros([M*N,1])
count=0
for i in range (0,M):
  t_old[count:count+N]=T[i]
  count=count+N

idx=np.random.permutation(M*N)

x=x_old[idx,:]
y=y_old[idx,:]
t=t_old[idx,:]


n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50
n_nodes_hl4 = 50
n_nodes_hl5 = 50
n_nodes_hl6 = 50



n_classes = 1
learn_rate = 0.0003
x_ph = tf.placeholder('float', [None, 1],name='input')
t_ph = tf.placeholder('float', [None, 1],name='input_time')
y_ph = tf.placeholder('float',[None,1],name='forcing')


hl_sigma = 0.02

# Routine to compute the neural network (5 hidden layers)
def neural_network_model(x_ph,t_ph):
    hidden_1_layer = {'weights': tf.Variable(name='w_h1',initial_value=tf.random_normal([2, n_nodes_hl1], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h1',initial_value=tf.random_normal([n_nodes_hl1], stddev=hl_sigma))}

    hidden_2_layer = {'weights': tf.Variable(name='w_h2',initial_value=tf.random_normal([n_nodes_hl1, n_nodes_hl2], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h2',initial_value=tf.random_normal([n_nodes_hl2], stddev=hl_sigma))}
                  
    
    hidden_3_layer = {'weights': tf.Variable(name='w_h3',initial_value=tf.random_normal([n_nodes_hl2, n_nodes_hl3], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h3',initial_value=tf.random_normal([n_nodes_hl3], stddev=hl_sigma))}
              
    hidden_4_layer = {'weights': tf.Variable(name='w_h4',initial_value=tf.random_normal([n_nodes_hl3, n_nodes_hl4], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h4',initial_value=tf.random_normal([n_nodes_hl4], stddev=hl_sigma))}    

                 
    hidden_5_layer = {'weights': tf.Variable(name='w_h5',initial_value=tf.random_normal([n_nodes_hl4, n_nodes_hl5], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_h5',initial_value=tf.random_normal([n_nodes_hl5], stddev=hl_sigma))} 

    output_layer = {'weights': tf.Variable(name='w_o',initial_value=tf.random_normal([n_nodes_hl4, n_classes], stddev=hl_sigma)),
                      'biases': tf.Variable(name='b_o',initial_value=tf.random_normal([n_classes], stddev=hl_sigma))}


    # (input_data * weights) + biases
    L1 = tf.add(tf.matmul(tf.concat([x_ph,t_ph],axis=1), hidden_1_layer['weights']), hidden_1_layer['biases'])
    L1 = tf.nn.tanh(L1)   

    L2 = tf.add(tf.matmul(L1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    L2 = tf.nn.tanh(L2)

    L3 = tf.add(tf.matmul(L2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    L3 = tf.nn.tanh(L3)
    
    L4 = tf.add(tf.matmul(L3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    L4 = tf.nn.tanh(L4)

    L5 = tf.add(tf.matmul(L4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    L5 = tf.nn.tanh(L5)
    output = tf.add(tf.matmul(L5, output_layer['weights']), output_layer['biases'], name='output')

    return output, L1, L2, L3, L4, L5

###### Main #################################


prediction, L1, L2, L3, L4, L5 = neural_network_model(x_ph,t_ph)
pred_dx = tf.gradients(prediction, x_ph)[0]
pred_dx2 = tf.gradients(tf.gradients(prediction, x_ph),x_ph)[0]
pred_dt=tf.gradients(prediction,t_ph)[0]
u=tf.math.sin(-(math.pi)*x_ph)+(1+x_ph)*(1-x_ph)*t_ph*prediction
#u=tf.math.sin(2*(math.pi)*x_ph)+x_ph*(1-x_ph)*t_ph*prediction
usq=tf.math.square(u)
usq_dx=tf.gradients(usq,x_ph)[0]
u_x=tf.gradients(u,x_ph)[0]
u_xx=tf.gradients(u_x,x_ph)[0]
u_t=tf.gradients(u,t_ph)[0]
    # Compute u and its second derivative
    #u = A + B*x_ph + (x_ph*x_ph)*prediction
    #dudx2 = (x_ph*x_ph)*pred_dx2 + 2.0*x_ph*pred_dx + 2.0*x_ph*pred_dx + 2.0*prediction
     
    # The cost function is just the residual of u''(x) - x*u(x) = 0, i.e. residual = u''(x)-x*u(x)
    #cost = tf.reduce_mean(tf.square(pred_dt- pred*pred_dx - pred_dx2 - y_ph))
cost=tf.reduce_mean(tf.square(u_t + u*u_x - (0.02)*u_xx - y_ph))
    #cost=tf.reduce_mean(tf.square(u_t - (0.01/math.pi)*u_xx - y_ph))

optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)


    # cycles feed forward + backprop
hm_epochs = 500

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Train in each epoch with the whole data
        for epoch in range(hm_epochs):
            #print('Epoch',epoch)
            epoch_loss = 0
            for step in range(0,M*N//batch_size):
              idx=step*batch_size
              inputX=x[idx:idx+batch_size,0].reshape([batch_size,1])
              inputY=y[idx:idx+batch_size,0].reshape([batch_size,1])
              inputT=t[idx:idx+batch_size,0].reshape([batch_size,1])
              #print(np.shape(inputX),np.shape(inputT),np.shape(inputY))   
              _, l = sess.run([optimizer,cost], feed_dict={x_ph:inputX, t_ph:inputT, y_ph:inputY})
              epoch_loss += l
            if epoch %10 == 0:
                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', float(epoch_loss/batch_size))


        # Predict a new input by adding a random number, to check whether the network has actually learned
        #x_valid = x + 0.0*np.random.normal(scale=0.1,size=(1))
 
           
        mypred =(sess.run(tf.squeeze(u),{x_ph:x_old,t_ph:t_old}))
        mypred_dx=(sess.run(tf.squeeze(u_x),{x_ph:x_old,t_ph:t_old}))
        #L1_np=(L1.eval())
        #L2_np=(L2.eval())
        #L3_np=(L3.eval())
        #L4_np=(L4.eval())
       
pred_reshape=np.reshape(mypred,[N,M],order='F')
pred_dx_reshape=np.reshape(mypred_dx,[N,M],order='F')

np.savetxt('prediction.csv',pred_reshape,delimiter=',')
np.savetxt('prediction_dx.csv',pred_dx_reshape,delimiter=',')
 
