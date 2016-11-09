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
		X, y = model.convert_to_tensors(testset)
		logits = model.DAE(X, is_training=False, noise=0.0)

		names_to_values, names_to_updates = model.error_metrics(logits, y)
		
		logdir = 'trainDAE.log'
		checkpoint_path = tf.train.latest_checkpoint(logdir)
		metric_values = slim.evaluation.evaluate_once(
			master='',
			checkpoint_path=checkpoint_path,
			logdir=logdir,
			eval_op=list(names_to_updates.values()),
			final_op=list(names_to_values.values()))

		names_to_values = dict(zip(list(names_to_values.keys()), metric_values))
		for name in names_to_values:
			print('%s: %f' % (name, names_to_values[name]))
