# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:27:36 2020

@author: Miki
"""

import os

import tensorflow as tf

import tfmodel

from os.path import basename

import imageio

from os import walk

import numpy as np

import matplotlib.pyplot as plt

#限制gpu使用
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

DATA_NAME = 'Data'
TRAIN_SOURCE = "Train"
TEST_SOURCE = 'Test'
RUN_NAME = "model_test"
OUTPUT_NAME = 'Output'
CHECKPOINT_FN = 'model.ckpt'


WORKING_DIR = os.getcwd() #取得此檔案路徑 C:\\Users\\Miki\\.spyder-py3\\SegNetCMR-master

TRAIN_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TRAIN_SOURCE) #train資料夾
TEST_DATA_DIR = os.path.join(WORKING_DIR, DATA_NAME, TEST_SOURCE) #test資料夾
###
ROOT_LOG_DIR = os.path.join(WORKING_DIR, OUTPUT_NAME) #SegNetCMR-master
LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_NAME)
CHECKPOINT_FL = os.path.join(LOG_DIR, CHECKPOINT_FN)

TRAIN_WRITER_DIR = os.path.join(LOG_DIR, TRAIN_SOURCE)
TEST_WRITER_DIR = os.path.join(LOG_DIR, TEST_SOURCE)
###
NUM_EPOCHS = 10
MAX_STEP = 2500
BATCH_SIZE = 1

LEARNING_RATE = 1e-04

SAVE_RESULTS_INTERVAL = 5
SAVE_CHECKPOINT_INTERVAL = 100


def get_image(file_name):

    image = imageio.imread(file_name) #(256, 256, 3)

    image = image[...,0][...,None]/255 #(256, 256, 1)
    image = image[None, ...] #(1, 256, 256, 256)
    
    return image

# 指定要列出所有檔案的目錄
route = "C:\\Users\\Miki\\.spyder-py3\\SegNetCMR-master\\Data\\Test\\Images\\Sunnybrook_Part3"
def read(path):
    files = []
    for root, dirs, file in walk(path):
        if '._' in file:
            print(file)
            break
        else:
            files.append(file)                       
    return files

test_image_route = read(route)
num_images = len(test_image_route[0]) #執行predict次數

def main():
    #train_data = tfmodel.GetData(TRAIN_DATA_DIR)
    #test_data = tfmodel.GetData(TEST_DATA_DIR)

    g = tf.Graph() #定義張量內容，可調用或保存

    with g.as_default(): #g

        #images, labels = tfmodel.placeholder_inputs(batch_size=BATCH_SIZE)
        
        images = tf.placeholder(tf.float32, [BATCH_SIZE, 256, 256, 1])

        logits, softmax_logits = tfmodel.inference(images, class_inc_bg=2) #return model answer

        #tfmodel.add_output_images(images=images, logits=logits, labels=labels)

        #loss = tfmodel.loss_calc(logits=logits, labels=labels)

        #global_step = tf.Variable(0, name='global_step', trainable=False)

        #train_op = tfmodel.training(loss=loss, learning_rate=1e-04, global_step=global_step)

        #accuracy = tfmodel.evaluation(logits=logits, labels=labels)

        #summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())

    sm = tf.train.SessionManager(graph=g)

    with sm.prepare_session("", init_op=init, saver=saver, checkpoint_dir=LOG_DIR) as sess:

        sess.run(tf.local_variables_initializer())

        #train_writer = tf.summary.FileWriter(TRAIN_WRITER_DIR, sess.graph)
        #test_writer = tf.summary.FileWriter(TEST_WRITER_DIR)

        #global_step_value, = sess.run([global_step])

        #print("Last trained iteration was: ", global_step_value)

        try:

           for i in range(num_images):
               
            file_name = os.path.join(route, test_image_route[0][i])
            test_data = get_image(file_name) #read image+normalization (1, 256, 256, 1) float64
            feed_dict = {images: test_data}
            labels_image = sess.run(logits, feed_dict=feed_dict)
            image = np.argmax(labels_image, axis=-1)[0, ...]
            plt.imsave(os.path.join(r'C:\Users\Miki\Desktop\SegNetCMR-master\Output\infer_test', basename(test_image_route[0][i])), image, cmap = 'gray')
                    


        except Exception as e:
            print('Exception')
            print(e)

        


if __name__ == '__main__':
    main()