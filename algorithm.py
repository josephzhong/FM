import numpy as np
import random
import tensorflow as tf


class FactorMachine:
    def __init__(self, w, rank=5, order=2):
        self.rank = rank
        self.order = order
        self.n_features = w
        self.init_std = 1.0
        self.x = tf.placeholder(tf.float32, name="x")
        self.y = tf.placeholder(tf.float32, name="y")
        self.w0 = tf.verify_tensor_all_finite(tf.Variable(0.0, name="bias"), msg='NaN or Inf in w0')
        w_weights = tf.random_uniform([self.n_features, 1], -self.init_std, self.init_std)
        # w_weights = tf.zeros([self.n_features, 1])
        self.w = tf.verify_tensor_all_finite(tf.Variable(w_weights, name="embedding_w"), msg='NaN or Inf in w')
        v_weights = tf.random_uniform([self.n_features, self.rank], -self.init_std, self.init_std)
        # v_weights = tf.zeros([self.n_features, self.rank])
        self.v = tf.verify_tensor_all_finite(tf.Variable(v_weights, name="embedding_v"), msg='NaN or Inf in v')
        self.dimension_1 = tf.tensordot(self.x, self.w, axes=[[1], [0]])
        self.dimension_2 = tf.multiply(tf.reduce_sum(tf.subtract(tf.pow(tf.tensordot(self.x, self.v, axes=[[1], [0]]), 2),
                                                     tf.tensordot(tf.pow(self.x, 2), tf.pow(self.v, 2), axes=[[1], [0]])),
                                                     axis=1, keepdims=True), 0.5)
        self.y_model_2d = tf.add(self.w0, tf.add(self.dimension_1, self.dimension_2))
        self.y_model = tf.reshape(self.y_model_2d, [-1])
        if self.order == 3:
            v3_weights = tf.random_uniform([self.n_features, self.rank], -self.init_std, self.init_std)
            self.v3 = tf.verify_tensor_all_finite(tf.Variable(v3_weights, name="embedding_v3"), msg='NaN or Inf in v')
            self.h3 = tf.pow(tf.tensordot(self.x, self.v3, axes=[[1], [0]]), 3)
            self.d21 = tf.multiply(tf.tensordot(tf.pow(self.x, 2), tf.pow(self.v3, 2), axes=[[1], [0]]),
                                   tf.tensordot(self.x, self.v3, axes=[[1], [0]]))
            self.d3 = tf.tensordot(tf.pow(self.x, 3), tf.pow(self.v3, 3), axes=[[1], [0]])
            self.dimension_3 = tf.reduce_sum(tf.multiply(tf.add(tf.subtract(self.h3, tf.multiply(self.d21, 3)),
                                                                tf.multiply(self.d3, 2)),
                                                         1/6),
                                             axis=1, keepdims=True)
            self.y_model_3d = tf.add(self.y_model_2d, self.dimension_3)
            self.y_model = tf.reshape(self.y_model_3d, [-1])
        self.loss_mse = tf.reduce_mean(tf.square(self.y - self.y_model))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.model = self.optimizer.minimize(self.loss_mse)
        self.train_loss_history = list()
        self.validate_loss_history = list()
        self.test_loss = 0.0
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        print("FM model initialized.")
        # self.summary_op = tf.summary.merge_all()

    def fit(self, data_x, data_y, validate_data_x, validate_data_y, test_data_x, test_data_y, batch_size=512, max_step=1000):
        for step in range(max_step):
            data_index = np.random.choice(range(len(data_x)), batch_size)
            batch_data_x = [data_x[i] for i in data_index]
            batch_data_y = [data_y[i] for i in data_index]
            self.session.run(self.model, feed_dict={self.x: batch_data_x,
                                                    self.y: batch_data_y})
            train_error = self.session.run(self.loss_mse, feed_dict={self.x: batch_data_x, self.y: batch_data_y})
            validate_data_index = np.random.choice(range(len(validate_data_x)), batch_size)
            batch_validate_data_x = [validate_data_x[i] for i in validate_data_index]
            batch_validate_data_y = [validate_data_y[i] for i in validate_data_index]
            validate_error = self.session.run(self.loss_mse, feed_dict={self.x: batch_validate_data_x,
                                                                        self.y: batch_validate_data_y})
            self.train_loss_history.append(train_error)
            self.validate_loss_history.append(validate_error)
            print(train_error, validate_error)
        test_loss = 0.0
        for index in range(0, len(test_data_x), batch_size):
            index_up_bound = index + batch_size if index + batch_size < len(test_data_x) else len(test_data_x)
            batch_test_data_x = test_data_x[index: index_up_bound]
            batch_test_data_y = test_data_y[index: index_up_bound]
            test_loss += self.session.run(self.loss_mse, feed_dict={self.x: batch_test_data_x, self.y: batch_test_data_y}) * len(batch_test_data_x)
        self.test_loss = test_loss / len(test_data_x)

    def predict(self, data_x):
        batch_size = 512
        result = list()
        for index in range(0, len(data_x), batch_size):
            index_up_bound = index + batch_size if index + batch_size < len(data_x) else len(data_x)
            batch_data_x = data_x[index: index_up_bound]
            # d1_value = self.session.run(self.dimension_1, feed_dict={self.x: batch_data_x})
            # d2_value = self.session.run(self.dimension_2, feed_dict={self.x: batch_data_x})
            # y_model_2d = self.session.run(self.y_model_2d, feed_dict={self.x: batch_data_x})
            if self.order == 3:
                h3_value = self.session.run(self.h3, feed_dict={self.x: batch_data_x})
                d21_value = self.session.run(self.d21, feed_dict={self.x: batch_data_x})
                d3_value = self.session.run(self.d3, feed_dict={self.x: batch_data_x})
                dimension_3_value = self.session.run(self.dimension_3, feed_dict={self.x: batch_data_x})
            value = self.session.run(self.y_model, feed_dict={self.x: batch_data_x})
            result.extend(value)
        return result

    def weights(self):
        if self.order == 2:
            return [self.w0.eval(session=self.session),
                    self.w.eval(session=self.session),
                    self.v.eval(session=self.session)]
        else:
            return [self.w0.eval(session=self.session),
                    self.w.eval(session=self.session),
                    self.v.eval(session=self.session),
                    self.v3.eval(session=self.session)]

    def load_state(self, path):
        self.saver.restore(self.session, path)

    def save_state(self, path):
        self.saver.save(self.session, path)

    def destroy(self):
        self.session.close()


if __name__ == "__main__":
    pass
    # with tf.device('/cpu:0'):
    #     x = tf.placeholder("float")
    #     y = tf.placeholder("float")
    #     w = tf.Variable([0.1] * 3)
    #     v = tf.Variable([[0.1] * 5] * 3)
    #     w0 = tf.Variable(0.22)
    #     dimension_1 = tf.tensordot(x, w, axes=[[1], [0]])
    #     dot_1 = tf.pow(tf.tensordot(x, v, axes=[[1], [0]]), 2)
    #     dimension_2 = tf.reduce_sum(tf.subtract(tf.pow(tf.tensordot(x, v, axes=[[1], [0]]), 2)
    #                                             , tf.tensordot(tf.pow(x, 2), tf.pow(v, 2), axes=[[1], [0]])), axis=1)
    #     y_model = tf.add(w0, tf.add(dimension_1, dimension_2))
    #     with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         y_model_value = sess.run(y_model, feed_dict={x: [[0.2, 0.4, 0.8], [0.1, 0.3, 0.5]], y: [0, 1]})
    #         print(y_model_value)
