import os, sys
import numpy as np
import tensorflow as tf
from random import shuffle
from model import build_model
from data_medley_db_reader import Medley
from data_classes import *

n_classes = vector_by_class.shape[0]

learning_rate = 0.000001#0.000001
max_epochs = 5000
max_steps = 0 #10000
batch_size = 100

DATA_PATH = "/media/ubuntu/DATA/MIR/MiniExperiment40k"
DATA_LIMIT = 0

validation_step = 10                   # validate the model each N steps
train_keep_probability = 0.8           # dropout

save_model = True                     # save the model
save_model_step = 500                 # save the mode each N steps
model_path = './model/model'          # where to save it

model_name = 'model-40k-2'           # this is used for tensoboard

if __name__ == "__main__":

	print(" (loading metadata...) ")
	dataset = Medley(DATA_PATH, limit=DATA_LIMIT)

	# read the mean, stddev and determine the data shape
	data_shape = dataset.get_input_data_shape()
	n_bands, n_samples   = data_shape

	with tf.Graph().as_default():
		g = tf.Graph()
		with g.as_default():

			X, Y, keep_prob, Y_pred, cost, optimizer, summary = build_model(n_bands, n_samples, n_classes, learning_rate, model_name=model_name)
		
			sess = tf.Session(graph=g)
			#train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
			#train_writer = tf.train.SummaryWriter('./summary/train', sess.graph)
			#validation_writer = tf.train.SummaryWriter('./summary/test')
			train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
			validation_writer = tf.summary.FileWriter('./summary/test')

			saver = tf.train.Saver()

			#init = tf.initialize_all_variables()
			init = tf.global_variables_initializer()
			sess.run(init)
			
			step = 0
			for epoch in range(max_epochs):
				print("epoch", epoch, "of", max_epochs)
				dataset.new_epoch()
				has_more_data = True
				while has_more_data:
					batch_xs, batch_ys, has_more_data = dataset.get_next_batch(batch_size)
					batch_xs = np.expand_dims(batch_xs, axis=3)

					calculated_cost, train_summary, _ = sess.run([cost, summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: train_keep_probability})
					train_writer.add_summary(train_summary, step)
					
					print("step", step, "batch ", batch_xs.shape, batch_ys.shape, "calculated_cost", calculated_cost)

					if step % validation_step == 0:
						batch_xs, batch_ys, _ = dataset.get_next_batch(batch_size, batch_type='test')
						batch_xs = np.expand_dims(batch_xs, axis=3)

						calculated_cost, validation_summary = sess.run([cost, summary], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
						validation_writer.add_summary(validation_summary, step)	
					
					if save_model and step % save_model_step == 0:
						save_path = saver.save(sess, model_path, global_step=step)
						print("Model saved in file: %s" % save_path)

					step = step + 1
					if max_steps > 0 and step >= max_steps:
						break

				if max_steps > 0 and step >= max_steps:
					break			
					

					
			
		


