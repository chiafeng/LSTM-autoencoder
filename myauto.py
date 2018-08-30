import datetime as dt

import numpy as np
import tensorflow as tf
import preprocess

tf.reset_default_graph()

learning_rate = 0.01
training_epochs = 300
display_step = 10          # how many steps to display a result

hidden_state_size = 100
samples = 100
time_steps = 50
step_dims = 681          # remove timestamp for test

# file path of training & testing data
regular_data_file = '~/Desktop/user_ml/tony_hsieh.1.csv'
compare_data_files = {
					  'AW': '~/Desktop/user_ml/amy_wu.1.csv',
					  'AC': '~/Desktop/user_ml/angus_chang.1.csv',
					  'BK': '~/Desktop/user_ml/ben_ko.1.csv',
					  'CH': '~/Desktop/user_ml/carl_huang.1.csv',
					  'JC': '~/Desktop/user_ml/jack_chang.1.csv',
					  'J': '~/Desktop/user_ml/jerry_wu.1.csv',
					  'KL': '~/Desktop/user_ml/kozue_lai.1.csv',
					  'MS': '~/Desktop/user_ml/michelle_song.1.csv',
					  #'TH': '~/Desktop/user_ml/tony_hsieh.1.csv',
					  'Y': '~/Desktop/user_ml/yiyi_lyu.1.csv',
					  'YF': '~/Desktop/user_ml/yvonne_feng.1.csv',
					  #'K': '~/Desktop/user_ml/kevin_chiu.1.csv',
					 }

save_dir = "save_{}/".format(dt.datetime.now().strftime("%m%d%H%M"))

# get 'samples' train_data and 3 test_data
data = preprocess.get_batch(regular_data_file, samples+3, time_steps, step_dims)
train_data = data[:samples]
test_data = data[samples:]

# get one data for other people
cmp_data = {}
for name, file in compare_data_files.items():
	cmp_data[name] = preprocess.get_batch(file, 1, time_steps, step_dims)[0]

##### Defining graph #####
# TODO: use shape [None, time_steps, step_dims]
seq_input = tf.placeholder(tf.float32, [1, time_steps, step_dims])

encoder_inputs = tf.unstack(seq_input)
print(len(encoder_inputs), encoder_inputs[0].shape)		#samples, (time_steps, step_dims)

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

# setup model saver and writer
saver = tf.train.Saver()
train_writer = tf.summary.FileWriter(save_dir+"train_plot", tf.get_default_graph())
test_writer = []
for i, data in enumerate(test_data):
	test_writer.append(tf.summary.FileWriter(save_dir+"test"+str(i+1), tf.get_default_graph()))

# setup writers of other people
cmp_writer = {}
for name in compare_data_files.keys():
	cmp_writer[name] = tf.summary.FileWriter(save_dir+name, tf.get_default_graph())

##### Start training #####
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print("start training....\nsave_dir={}\nlearning rate: {}\n".format(save_dir, learning_rate))
	x = train_data
	for epoch in range(training_epochs):
		# TODO: always use same batch
		# circular repeat the training data
		feed = {seq_input: x[epoch%samples]}
		_, cost_value, summary, out = sess.run([optimizer, loss, merged_summary, dec_outputs], feed_dict=feed)
		if epoch % display_step == 0:

			print(len(out), out[0].shape)

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
			for i, data in enumerate(test_data):
				feed = {seq_input: data}
				cost_value, summary = sess.run([loss, merged_summary], feed_dict=feed)
				test_writer[i].add_summary(summary, epoch)

			# calculate loss for cmp_data
			for name, data in cmp_data.items():
				feed = {seq_input: data}
				cost_value, summary = sess.run([loss, merged_summary], feed_dict=feed)
				cmp_writer[name].add_summary(summary, epoch)

print("Optimization Finished!")

train_writer.close()
for writer in test_writer:
	writer.close()
for writer in cmp_writer.values():
	writer.close()