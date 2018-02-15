from keras.callbacks import TensorBoard
from keras.layers import Activation, BatchNormalization, Conv2D, Lambda
from keras import backend as K
import numpy as np
import cv2
import tensorflow as tf

CHANNEL_AXIS = 1 if K.image_data_format() == "channels_first" else -1


def Conv2D_BN(X, filters, kernel_size, strides, padding, name, activation="relu", use_bn=True):
    """Wrapper for creating Conv2D + BatchNormalization (optional) + Activation (optional)

    :param X: input tensor [None, width, height, channels]
    :param filters: number of filters
    :param kernel_size: kernel size, same format as in Keras Conv2D
    :param strides: stride, same format as in Keras Conv2D
    :param padding: padding, same format as in Keras Conv2D
    :param name: layer name
    :param activation: activation function (in the format acceptable to Activation class in Keras), or None for no
        activation function (== linear, f(x) = x). Default: "relu"
    :param use_bn: whether to use batch normalization
    :return: output tensor
    """
    X = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(X)
    if use_bn:
        bn_name = name + "_BN"
        X = BatchNormalization(axis=CHANNEL_AXIS, scale=False, name=bn_name)(X)
    if activation is not None:
        ac_name = name + "_" + activation
        X = Activation("relu", name=ac_name)(X)
    return X

def shortcut_summation(X, X_shortcut, scale, name):
    """Summation of shortcut and inception tensors

    :param X: post-inception tensor, [None, width, height, channels]
    :param X_shortcut: shortcut tensor, [None, width, height, channels]
    :param scale: scale factor. Suggested values in article are between 0.1 and 0.3
    :param name: layer name
    :return: output tensor
    """
    X = Lambda(lambda inputs, scale: inputs[0] * scale + inputs[1],
               output_shape=K.int_shape(X)[1:],
               arguments={"scale": scale},
               name=name)([X, X_shortcut])
    return X

def contrastive_loss(y_true, y_pred, margin=0.5):
    """Loss function for minimizing distance between similar images and maximizing between dissimilar
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param y_true: ground truth labels, 0 if images similar and 1 if dissimilar
    :param y_pred: predicted labels
    :param margin: maximum distance between dissimilar images at which they are still considered for loss calculations
    """
    left_output = y_pred[0::2, :]
    right_output = y_pred[1::2, :]
    dist = tf.reduce_sum(tf.square(left_output - right_output), 1)
    dist_sqrt = tf.sqrt(dist)
    L_s = dist
    L_d = tf.square(tf.maximum(0.0, margin - dist_sqrt))
    loss = y_true * L_d + (1 - y_true) * L_s
    return loss

def read_image(image_path, image_size):
    """Image readings, resizing and normalization

    :param image_path: path to the image file
    :param image_size: new image height and width as a tuple/list
    :return: image as np.array
    """
    img = cv2.imread(image_path)
    img_res = cv2.resize(img, image_size)
    img_rgb = img_res[..., ::-1]
    img_norm = np.around(img_rgb / 255.0, decimals=12)
    return img_norm

def img_to_embedding(model, image_path, image_size):
    """Get embedding for one image

    :param model: trained model
    :param image_path: path to the image file
    :param image_size: new image height and width as a tuple/list
    :return: embedding
    """
    img = read_image(image_path, image_size)
    return model.predict_on_batch(img)

def batch_to_embedding(model, files, image_size):
    """Get embedding for the batch of the images

    :param model: trained model
    :param files: list of paths to image files
    :param image_size: new image height and width as a tuple/list
    :return: embeddings
    """
    X = np.ndarray(shape=[len(files), image_size[0], image_size[1], 3])
    for idx, image in enumerate(files):
        img_norm = read_image(image, image_size)
        X[idx] = img_norm
    embeddings = model.predict_on_batch(X)
    return embeddings

class TensorBoardWrapper(TensorBoard):
    """Wrapper fo TensorBoard callback for correct work with generators. Updates self.validation_data
    https://github.com/keras-team/keras/issues/3358
    """

    def __init__(self, batch_gen, nb_steps, **kwargs):
        """

        :param batch_gen: generator
        :param nb_steps: number of batches yielded from generator before the epoch end
        :param kwargs: TensorBoard kwargs
        """
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen
        self.nb_steps = nb_steps

    def on_epoch_end(self, epoch, logs):
        """Redefining function for correct update of self.validation_data

        :param epoch: epoch number
        :param logs: logs
        :return:
        """
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)