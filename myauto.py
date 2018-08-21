import datetime as dt

import numpy as np
import tensorflow as tf
import preprocess

tf.reset_default_graph()

learning_rate = 0.01
training_epochs = 10000
display_step = 100          # how many steps to display a result

hidden_state_size = 100
samples = 1
time_steps = 200 
step_dims = 1894          # remove timestamp for test

save_dir = "/Users/feng/Desktop/ML/Tensorflow-seq2seq-autoencoder/save_{}/"\
           .format(dt.datetime.now().strftime("%m%d%H%M"))

data = preprocess.get_batch('vector/angus_chang_201807.data.angus_chang',
                            time_steps*2, step_dims)
# slice data from 2nd dimension
train_data = data[:, :time_steps, :]
test_data = data[:, time_steps:time_steps*2, :]

J_data = preprocess.get_batch('vector/jerry_wu_201807.data.jerry_wu',
                              time_steps, step_dims)
# K_data = preprocess.get_batch('vector/kevin_chiu_201807.data.kevin_chiu',
#                               time_steps, step_dims)
Y_data = preprocess.get_batch('vector/yiyi_lyu_201807.data.yiyi_lyu',
                              time_steps, step_dims)

##### Defining graph #####
seq_input = tf.placeholder(tf.float32, [samples, time_steps, step_dims])

encoder_inputs = tf.unstack(seq_input)

layers = [tf.contrib.rnn.BasicLSTMCell(units) for units in [hidden_state_size, step_dims]]
enc_cell = tf.nn.rnn_cell.MultiRNNCell(layers)
enc_cell = tf.contrib.rnn.OutputProjectionWrapper(enc_cell, step_dims)
dec_outputs, dec_state = tf.contrib.rnn.static_rnn(enc_cell, encoder_inputs, dtype=tf.float32)

y_true = encoder_inputs
y_pred = dec_outputs

loss = 0
for i in range(len(y_true)):
       loss += tf.reduce_sum(tf.square(tf.subtract(y_pred[i], y_true[i])))
# summary for FileWriter
loss_summary = tf.summary.scalar("loss", loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

merged_summary = tf.summary.merge_all()

##### Start training #####
saver = tf.train.Saver()
train_writer = tf.summary.FileWriter(save_dir+"train_plot", tf.get_default_graph())
test_writer = tf.summary.FileWriter(save_dir+"test_plot", tf.get_default_graph())
J_writer = tf.summary.FileWriter(save_dir+"J", tf.get_default_graph())
K_writer = tf.summary.FileWriter(save_dir+"K", tf.get_default_graph())
Y_writer = tf.summary.FileWriter(save_dir+"Y", tf.get_default_graph())

with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())

       print("start training....\nsave_dir={}\nlearning rate: {}\n".format(save_dir, learning_rate))
       x = train_data
       for epoch in range(training_epochs):
           feed = {seq_input: x}
           _, cost_value, summary = sess.run([optimizer, loss, merged_summary], feed_dict=feed)
           if epoch % display_step == 0:
               print ("logits")
               a = sess.run(y_pred, feed_dict=feed)
               print (a)
               print ("labels")
               b = sess.run(y_true, feed_dict=feed)
               print (b)

               print("Epoch:" + '%04d' % (epoch+1), "cost=" + "{:.9f}".format(cost_value))

               train_writer.add_summary(summary, epoch)

               saver.save(sess, save_dir + "model.ckpt")

               # calculate loss for test data
               feed = {seq_input: test_data}
               cost_value, summary = sess.run([loss, merged_summary], feed_dict=feed)
               test_writer.add_summary(summary, epoch)

               # calculate loss for J
               feed = {seq_input: J_data}
               cost_value, summary = sess.run([loss, merged_summary], feed_dict=feed)
               J_writer.add_summary(summary, epoch)

               # calculate loss for K
               # feed = {seq_input: K_data}
               # cost_value, summary = sess.run([loss, merged_summary], feed_dict=feed)
               # K_writer.add_summary(summary, epoch)

               # calculate loss for Y
               feed = {seq_input: Y_data}
               cost_value, summary = sess.run([loss, merged_summary], feed_dict=feed)
               Y_writer.add_summary(summary, epoch)

print("Optimization Finished!")
train_writer.close()
test_writer.close()
J_writer.close()
K_writer.close()
Y_writer.close()