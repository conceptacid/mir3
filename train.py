import os, sys
import numpy as np
import tensorflow as tf
from random import shuffle
from model import build_model
from data_medley_db_reader import Medley
from data_classes import *

n_classes = vector_by_class.shape[0]

learning_rate = 0.000001
max_epochs = 5000
max_steps = 0 #10000
batch_size = 100

DATA_PATH = "/media/ubuntu/DATA/MIR/MiniExperiment40k"
DATA_LIMIT = 0

validation_step = 10                   # validate the model each N steps
train_keep_probability = 0.8           # dropout

model_name = 'm40'           # this is used for tensoboard

train_model = True
save_model  = True                             # save the model
save_model_step = 300                          # save the mode each N steps
model_path = './model/' + model_name           # where to save it
init_step = 0 
#init_load_model_from = './model/doesitworknow-1000' # model_path + "-1000" #./model/m5-' + str(init_step)
init_load_model_from = ''


if __name__ == "__main__":

	print(" (loading metadata...) ")
	dataset = Medley(DATA_PATH, limit=DATA_LIMIT)

	# read the mean, stddev and determine the data shape
	data_shape = dataset.get_input_data_shape()
	n_bands, n_samples   = data_shape


	g = tf.Graph()
	with g.as_default():

	
		with tf.Session(graph=g) as sess:

			X, Y, keep_prob, Y_pred, cost, optimizer, summary, accuracy, w4 = build_model(n_bands, n_samples, n_classes, learning_rate, model_name=model_name)

			init = tf.global_variables_initializer()
			sess.run(init)

			train_writer = tf.summary.FileWriter('./summary/train', sess.graph)
			validation_writer = tf.summary.FileWriter('./summary/test')

			#list_of_variables_to_restore = tf.contrib.framework.get_variables_to_restore()
			#print("Restore variables: ", [var.name for var in list_of_variables_to_restore])

			'''
			with tf.variable_scope("fc_layer_4", reuse=True):	
				w = tf.get_variable("W")  # The same as v above.
				w = w.eval()
				print("DEBUG w.shape", w.shape, w[0,0], w[100,200], w[100,1024], w[1900,1100], w[1,10])


			print("W4 name = ", w4.name)
			xxx = sess.run(w4)
			print("DEBUG w4.shape", xxx.shape, xxx[0,0], xxx[100,200], xxx[100,1024], xxx[1900,1100], xxx[1,10])
			'''

			saver = tf.train.Saver()
			if len(init_load_model_from) > 0:
				print("LOADING MODEL FROM", init_load_model_from)
				saver.restore(sess, init_load_model_from)
				'''
				xxx = sess.run(w4)
				print("DEBUG w4.shape", xxx.shape, xxx[0,0], xxx[100,200], xxx[100,1024], xxx[1900,1100], xxx[1,10])					
				with tf.variable_scope("fc_layer_4", reuse=True):
					w = tf.get_variable("W")  # The same as v above.
					w = w.eval()
					print("DEBUG w.shape", w.shape, w[0,0], w[100,200], w[100,1024], w[1900,1100], w[1,10])
				#pass
				'''
			else:	
				print("INITIALIZING MODEL (starting training from scratch)")
				#print(sess.run(tf.all_variables()))
				#init = tf.initialize_all_variables()
				'''
				with tf.variable_scope("fc_layer_4", reuse=True):
					w = tf.get_variable("W")  # The same as v above.
					w = w.eval()
					print("DEBUG w.shape", w.shape, w[0,0], w[100,200], w[100,1024], w[1900,1100], w[1,10])
				xxx = sess.run(w4)
				print("DEBUG w4.shape", xxx.shape, xxx[0,0], xxx[100,200], xxx[100,1024], xxx[1900,1100], xxx[1,10])					
				'''
				
				
		
			
			step = init_step + 1
			for epoch in range(max_epochs):
				print("epoch", epoch, "of", max_epochs)
				dataset.new_epoch()
				has_more_data = True
				while has_more_data:
					
					if train_model:
						batch_xs, batch_ys, has_more_data = dataset.get_next_batch(batch_size)
						batch_xs = np.expand_dims(batch_xs, axis=3)

						calculated_cost, train_summary, _ = sess.run([cost, summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: train_keep_probability})
						train_writer.add_summary(train_summary, step)
					
						print("Training epoch", epoch+1, "step", step, "batch ", batch_xs.shape, batch_ys.shape, "calculated_cost", calculated_cost)

					if not train_model or step % validation_step == 0:
						batch_xs, batch_ys, _  = dataset.get_next_batch(batch_size, batch_type='test')
						batch_xs = np.expand_dims(batch_xs, axis=3)

						calculated_cost, validation_summary, calculated_accuracy = sess.run([cost, summary, accuracy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
						if train_model:
							validation_writer.add_summary(validation_summary, step)	

						print("Validation step", step, "batch",  "cost", calculated_cost, "accuracy", calculated_accuracy)	


						batch_xs, batch_ys, _ = dataset.get_special_batch()
						batch_xs = np.expand_dims(batch_xs, axis=3)
						special_val = sess.run([Y_pred], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})[0]

						print("Special step", batch_ys, "predicted=", special_val)	
					
					if train_model and save_model and step % save_model_step == 0:
						save_path = saver.save(sess, model_path, global_step=step)
						print("Model saved in file: %s" % save_path)

						
						'''
						with tf.variable_scope("fc_layer_4", reuse=True):
							w = tf.get_variable("W")  # The same as v above.
							w = w.eval()
							print("DEBUG w.shape", w.shape, w[0,0], w[100,200], w[100,1024], w[1900,1100], w[1,10])
						xxx = sess.run(w4)
						print("DEBUG w4.shape", xxx.shape, xxx[0,0], xxx[100,200], xxx[100,1024], xxx[1900,1100], xxx[1,10])
						'''

					step = step + 1
					if max_steps > 0 and step >= max_steps:
						break

				if max_steps > 0 and step >= max_steps:
					break			
					

					
				
			


