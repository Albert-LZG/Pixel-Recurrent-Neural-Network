import os
import logging

import numpy as np
from tqdm import trange
import tensorflow as tf

from utils import *
from network import Network
from statistic import Statistic

################################# CONFIGURATION #################################
flags = tf.app.flags
# Network  
flags.DEFINE_string("model", "pixel_rnn", "type of the network [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("hidden_dims", 64, "dimension of hidden state for LSTM or conv layers")
flags.DEFINE_integer("out_hidden_dims", 64, "dimension of ouput hidden state")
flags.DEFINE_integer("inner_recurrent_stack_length", 2, "length of staked recurrent layers")
flags.DEFINE_integer("out_layer_stack_length", 2, "length of output layers")
flags.DEFINE_boolean("use_residual", True, "whether to use residual or not")

# Training 
flags.DEFINE_float("epoch", 100000, "length of epoch")
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value for gradient clipping")
flags.DEFINE_float("save_at", 5, "number of epochs for saving model")
flags.DEFINE_float("test_at", 1, "number of epochs for testing")
flags.DEFINE_float("sample_at", 1, "number of epochs sampling")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu")
flags.DEFINE_string("model_name", "", "name of the model to resume training")
flags.DEFINE_boolean("use_dynamic_rnn", False, "whether to use tf.nn.dynamicrnn")
flags.DEFINE_string("optimizer", "ADAM", "name of the optimizer(RMS, ADAM, ADADELTA)")

# Data
flags.DEFINE_string("data", "mnist", "name of the dataset [mnist, cifar]")
flags.DEFINE_string("data_dir", "data", "name of the data directory")
flags.DEFINE_string("sample_dir", "samples", "name of the sample directory]")

# Options
flags.DEFINE_boolean("is_train", True, "Training or testing")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 1202, "random seed for python")
flags.DEFINE_boolean("display", False, "random seed for python")

conf = flags.FLAGS

################################## Logging ##################################
logger = logging.getLogger()
formatter = logging.Formatter(fmt = "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s",
                              datefmt = "%y-%m-%d %H:%M:%S")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel(conf.log_level)

##################################   main   ##################################
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)

def main(_):
    # 0. Manage directories: data_dir-where to get data, sample_dir-where to save data
    model_dir = get_model_dir(config = conf, exceptions = [
        'batch_size', 'epoch', 'save_at', 'test_at', 'use_gpu','use_dynamic_rnn',
        'data_dir', 'sample_dir', 'is_train', 'log_level', 'random_seed', 'display', 'sample_at'])

    preprocess_conf(conf)

    DATA_DIR = os.path.join(conf.data_dir, conf.data)
    SAMPLE_DIR = os.path.join(conf.sample_dir, conf.data, model_dir)

    check_and_create_dir(DATA_DIR)
    check_and_create_dir(SAMPLE_DIR)

    # 1. Prepare dataset
    if conf.data == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(DATA_DIR, one_hot = True)

        next_train_batch = lambda x: mnist.train.next_batch(x)[0]
        next_test_batch = lambda x: mnist.test.next_batch(x)[0]

        height, width, channel = 28, 28, 1

        train_step = mnist.train.num_examples / conf.batch_size
        test_step = mnist.test.num_examples / conf.batch_size

    with tf.Session() as sess:
        network = Network(sess, conf, height, width, channel)
        stat = Statistic(sess, conf.data, model_dir, tf.trainable_variables(), conf.test_at)
        
        if conf.model_name != None:
            stat.load_model()
            
            if conf.is_train:
                logger.info("Training Starts!")

                it_epoch = trange(conf.epoch, ncols = 70, initial = 0, desc = 'Epoch')
                for epoch in it_epoch:

                    lr = max(conf.learning_rate * (0.99 ** epoch), 0.000001)
                    #logger.info("[Epoch %d] leraning_rate: %f" % (epoch, lr))

                    # 2. Train
                    train_iterator = trange(train_step, ncols = 70, initial = 0, desc = 'Train')
                    total_train_costs = []
                    for idx in train_iterator:
                        if conf.data == "mnist":
                            images = binarize(next_train_batch(conf.batch_size))\
                                .reshape([conf.batch_size, height, width, channel])
                        else: # cifar
                            pass
                        
                        cost = network.train(images, lr)
                        total_train_costs.append(cost)
                        train_iterator.set_description("[Epoch %d | batch %d] lr: %f / train loss: %.3f"
                                                       % (epoch, idx, lr, cost))
                    
                    # 3. Test
                    if epoch % conf.test_at == 0:
                        total_test_costs = []
                        test_iterator = trange(test_step, ncols = 70, initial = 0, desc = 'Test')
                        for idx in test_iterator:
                            if conf.data ==  "mnist":
                                images = binarize(next_test_batch(conf.batch_size))\
                                    .reshape([conf.batch_size, height, width, channel])
                            else:
                                pass
                            
                            outputs, cost = network.test(images)
                            total_test_costs.append(cost)
                            train_iterator.set_description("[Epoch %d | batch %d] test loss: %.3f"
                                                           % (epoch, idx, cost))

                        # save test images
                        save_images(outputs[:100, :, :, :], height, width, 10, 10,
                                    directory = SAMPLE_DIR, prefix = "epoch_%d_cost_%f" % (epoch, cost))
                        avg_train_cost = np.mean(total_train_costs)
                        avg_test_cost = np.mean(total_test_costs)

                    if epoch % conf.save_at == 0:
                        stat.on_step(avg_train_cost, avg_test_cost)
                    
                    # 4. Generate Sample
                    if epoch % conf.sample_at == 0:
                        samples = network.generate()
                        save_images(samples, height, width, 10, 10,
                                    directory = SAMPLE_DIR, prefix = "epoch_%s" % epoch)

                    it_epoch.set_description("train l: %.3f, test l: %.3f"
                                     % (avg_train_cost, avg_test_cost))
        
            else:
                logger.info("Generating Image..")
                
                samples = network.generate()
                save_images(samples, height, width, 10, 10, directory = SAMPLE_DIR)
if __name__ == "__main__":
    tf.app.run()
