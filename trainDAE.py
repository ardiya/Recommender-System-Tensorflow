from dataset import load_movielens_100k, load_movielens_1m
import model
import tensorflow as tf
slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		trainset, testset = load_movielens_1m()
		X, y = model.convert_to_tensors(trainset)
		logits = model.DAE(X,noise=0.1)
		loss = model.loss(logits, y)
		train_op = model.train(loss)

		final_loss =slim.learning.train(
			train_op,
			logdir='trainDAE.log',
			number_of_steps=1000,
			save_summaries_secs=5
		)
