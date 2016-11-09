from dataset import load_movielens_100k, load_movielens_1m, to_tensor
import model
import tensorflow as tf
slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == '__main__':
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		trainset, testset = load_movielens_1m()
		X = to_tensor(trainset)
		logits, loss = model.AE(X)
		train_op = model.train(loss)

		final_loss =slim.learning.train(
			train_op,
			logdir='trainAE.log',
			number_of_steps=5000,
			save_summaries_secs=5
		)
