import cv2
import os
import random
import tensorflow as tf
import numpy as np
 
class Sample:
    def __init__(self):
        self.x = []
        self.y = []
    def get_batch(self,n):
        imgs = os.listdir('dataset')
        for i in range(n):
            index = random.randint(0,len(imgs)-1)
            img = imgs[index]
            self.x.append(cv2.imread('dataset\{}'.format(img)))
            position = img.split('.')
            x1 = int(position[1])/224
            y1 = int(position[2])/224
            x2 = int(position[3])/224
            y2 = int(position[4])/224
            label = int(position[5])
            self.y.append([x1,y1,x2,y2,label])
        self.x1 = np.array(self.x)
        self.x1 = (self.x1/255-0.5)*2
        return self.x,self.x1,np.array(self.y)
 
class Net:
    def __init__(self):
        self.x = tf.placeholder(shape=[None,224,224,3],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,5],dtype=tf.float32)
        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3,3,3,64],dtype=tf.float32,stddev=tf.sqrt(2/(3*3*3))))
        self.conv1_b = tf.Variable(tf.zeros([64]))
        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3,64,128], dtype=tf.float32, stddev=tf.sqrt(2 / (3*3*64))))
        self.conv2_b = tf.Variable(tf.zeros([128]))
        self.conv3_w = tf.Variable(tf.truncated_normal(shape=[3, 3,128,256], dtype=tf.float32, stddev=tf.sqrt(2 / (3*3*128))))
        self.conv3_b = tf.Variable(tf.zeros([256]))
        self.conv4_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 256,256], dtype=tf.float32, stddev=tf.sqrt(2 / (3*3*256))))
        self.conv4_b = tf.Variable(tf.zeros([256]))
        self.conv5_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 256,512], dtype=tf.float32, stddev=tf.sqrt(2 / (3*3*256))))
        self.conv5_b = tf.Variable(tf.zeros([512]))
        self.w1 = tf.Variable(tf.truncated_normal(shape=[7*7*512,512],dtype=tf.float32,stddev=tf.sqrt(1/(7*7*512))))
        self.b1 = tf.Variable(tf.zeros([512]))
        self.w2 = tf.Variable(tf.truncated_normal(shape=[512,256], dtype=tf.float32,stddev=tf.sqrt(1/512)))
        self.b2 = tf.Variable(tf.zeros([256]))
        self.w3_1 = tf.Variable(tf.truncated_normal(shape=[256,4], dtype=tf.float32,stddev=tf.sqrt(1/256)))
        self.b3_1 = tf.Variable(tf.zeros([4]))
        self.w3_2 = tf.Variable(tf.truncated_normal(shape=[256, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.b3_2 = tf.Variable(tf.zeros([1]))
    def forward(self):
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.x,self.conv1_w,strides=[1,1,1,1],padding='SAME')+self.conv1_b))
        self.pool1 = tf.nn.relu(tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID'))#112
        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b))
        self.pool2 = tf.nn.relu(tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))  # 56
        self.conv3 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.pool2, self.conv3_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b))
        self.pool3 = tf.nn.relu(tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))#28
        self.conv4 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.pool3, self.conv4_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b))
        self.pool4 = tf.nn.relu(tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))#14
        self.conv5 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.pool4, self.conv5_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b))
        self.pool5 = tf.nn.relu(tf.nn.max_pool(self.conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))#7
        self.flat = tf.reshape(self.pool5,[-1,7*7*512])
        self.f1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.flat,self.w1)+self.b1))
        self.f2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(self.f1, self.w2) + self.b2))
        self.out_1 = tf.matmul(self.f2, self.w3_1) + self.b3_1
        self.out_2 = tf.matmul(self.f2, self.w3_2) + self.b3_2
    def backward(self):
        loss1 = tf.reduce_mean((self.out_1-self.y[:,:4])**2)#bbox损失
        labels = tf.reshape(self.y[:, 4],[-1,1])
        loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.nn.sigmoid(self.out_2), labels=labels))#置信度损失
        self.loss = loss1+loss2
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
 
if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1000):
            sample = Sample()
            img,x, y = sample.get_batch(50)
            loss,_,out,Confidence = sess.run([net.loss,net.optimizer,net.out_1,net.out_2],feed_dict={net.x:x,net.y:y})
            position = [abs(int(n)) for n in list(out[0] * 224)]
            print(loss)
            if (epoch+1)%500 == 0:
                saver.save(sess,save_path='params\chpk')
