from keras.layers import MaxPool2D, Concatenate, Input, Lambda, Activation, GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras import backend as K
from keras.utils.vis_utils import plot_model

from helpers import Conv2D_BN, shortcut_summation, CHANNEL_AXIS, contrastive_loss

class InceptionResnetV2:
    """Inception-ResNet-v2 initialization
    Stem -> 5 x Inception-ResNet-A -> Reduction-A -> 10 x Inception-ResNet-B -> Reduction-B -> 5x Inception-ResNet-C ->
    GlobalAveragePooling -> Dropout -> Dense(embedding_shape)
    """
    def __init__(self, input_shape=(299, 299, 3), dropout_keep_prob=0.8, embedding_shape=128):
        """
        :param input_shape: input image shape
        :param dropout_keep_prob: dropout keep probability
        :param embedding_shape: shape of resulting embedding
        """
        self.input_shape = input_shape
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_shape = embedding_shape

    def stem(self, X):
        """
        :param X: input tensor
        :return: output tensor
        """
        # input X: 299 x 299 x 3
        X = Conv2D_BN(X, filters=32, kernel_size=3, strides=2, padding="valid", name="Stem_1_conv2d")
        # 149 x 149 x 32
        X = Conv2D_BN(X, filters=32, kernel_size=3, strides=1, padding="valid", name="Stem_2_conv2d")
        # 147 x 147 x 32
        X = Conv2D_BN(X, filters=64, kernel_size=3, strides=1, padding="same", name="Stem_3_conv2d")
        # 147 x 147 x 64
        branch_1 = MaxPool2D(pool_size=3, strides=2, padding="valid", name="Stem_4_pool_branch_1_1")(X)
        branch_2 = Conv2D_BN(X, filters=96, kernel_size=3, strides=2, padding="valid",
                             name="Stem_4_conv2d_branch_2_1")
        # 73 x 73 x 160 (73 x 73 x 64 + 73 x 73 x 96)
        X = Concatenate(axis=CHANNEL_AXIS, name="Stem_4_concatenate")([branch_1, branch_2])
        # branch 1
        branch_1 = Conv2D_BN(X, filters=64, kernel_size=1, strides=1, padding="same",
                             name="Stem_5_conv2d_branch_1_1")
        branch_1 = Conv2D_BN(branch_1, filters=96, kernel_size=3, strides=1, padding="valid",
                             name="Stem_5_conv2d_branch_1_2")
        # branch 2
        branch_2 = Conv2D_BN(X, filters=64, kernel_size=1, strides=1, padding="same",
                             name="Stem_5_conv2d_branch_2_1")
        branch_2 = Conv2D_BN(branch_2, filters=64, kernel_size=(7, 1), strides=1, padding="same",
                             name="Stem_5_conv2d_branch_2_2")
        branch_2 = Conv2D_BN(branch_2, filters=64, kernel_size=(1, 7), strides=1, padding="same",
                             name="Stem_5_conv2d_branch_2_3")
        branch_2 = Conv2D_BN(branch_2, filters=96, kernel_size=3, strides=1, padding="valid",
                             name="Stem_5_conv2d_branch_2_4")
        X = Concatenate(axis=CHANNEL_AXIS, name="Stem_5_concatenate")([branch_1, branch_2])
        # 71 x 71 x 192 (71 x 71 x 96 + 71 x 71 x 96)
        branch_1 = Conv2D_BN(X, filters=192, kernel_size=3, strides=2, padding="valid", name="Stem_6_conv2d_branch_1_1")
        branch_2 = MaxPool2D(pool_size=3, strides=2, padding="valid", name="Stem_6_conv2d_branch_2_1")(X)
        X = Concatenate(axis=CHANNEL_AXIS, name="Stem_6_concatenate")([branch_1, branch_2])
        # 35 x 35 x 384 (35 x 35 x 192 + 35 x 35 x 192)
        return X

    def resnet_A(self, X, scale=0.17, iteration=None):
        """First Inception-Resnet block Inception-ResNet-A, 35x35 (given the default input size is used)

        :param X: input tensor
        :param scale: scaling factor for output tensor
        :param iteration: iteration index (required for layer name generation)
        :return: output tensor
        """
        X_shortcut = X
        # branch 1
        branch_1 = Conv2D_BN(X, filters=32, kernel_size=1, strides=1, padding="same",
                             name="Resnet_A_{}_conv2d_branch_1_1".format(iteration))
        # branch 2
        branch_2 = Conv2D_BN(X, filters=32, kernel_size=1, strides=1, padding="same",
                             name="Resnet_A_{}_conv2d_branch_2_1".format(iteration))
        branch_2 = Conv2D_BN(branch_2, filters=32, kernel_size=3, strides=1, padding="same",
                             name="Resnet_A_{}_conv2d_branch_2_2".format(iteration))
        # branch 3
        branch_3 = Conv2D_BN(X, filters=32, kernel_size=1, strides=1, padding="same",
                             name="Resnet_A_{}_conv2d_branch_3_1".format(iteration))
        branch_3 = Conv2D_BN(branch_3, filters=48, kernel_size=3, strides=1, padding="same",
                             name="Resnet_A_{}_conv2d_branch_3_2".format(iteration))
        branch_3 = Conv2D_BN(branch_3, filters=64, kernel_size=3, strides=1, padding="same",
                             name="Resnet_A_{}_conv2d_branch_3_3".format(iteration))
        branches = [branch_1, branch_2, branch_3]
        X = Concatenate(axis=CHANNEL_AXIS, name="Resnet_A_{}_concatenate".format(iteration))(branches)
        X = Conv2D_BN(X, filters=384, kernel_size=1, strides=1, padding="same",
                      name="Resnet_A_{}_conv2d_final".format(iteration), activation=None, use_bn=False)
        # Shortcut connection
        X = shortcut_summation(X, X_shortcut, scale, name="Resnet_A_{}_shortcut_sum".format(iteration))
        X = Activation("relu", name="Resnet_A_{}_relu".format(iteration))(X)
        return X

    def reduction_A(self, X):
        """Reduction-A block

        :param X: input tensor
        :return: output tensor
        """
        # branch 1
        branch_1 = MaxPool2D(pool_size=3, strides=2, padding="valid", name="Reduction_A_pool_branch_1")(X)
        # branch 2
        branch_2 = Conv2D_BN(X, filters=384, kernel_size=3, strides=2, padding="valid",
                             name="Reduction_A_conv2d_branch_2_1")
        # branch 3
        branch_3 = Conv2D_BN(X, filters=256, kernel_size=1, strides=1, padding="same",
                             name="Reduction_A_conv2d_branch_3_1")
        branch_3 = Conv2D_BN(branch_3, filters=256, kernel_size=3, strides=1, padding="same",
                             name="Reduction_A_conv2d_branch_3_2")
        branch_3 = Conv2D_BN(branch_3, filters=384, kernel_size=3, strides=2, padding="valid",
                             name="Reduction_A_conv2d_branch_3_3")
        branches = [branch_1, branch_2, branch_3]
        X = Concatenate(axis=CHANNEL_AXIS, name="Reduction_A_concatenate")(branches)
        return X

    def resnet_B(self, X, scale=0.1, iteration=None):
        """Second Inception-Resnet block Inception-ResNet-B, 17x17 (given the default input size is used)

        :param X: input tensor
        :param scale: scaling factor for output tensor
        :param iteration: iteration index (required for layer name generation)
        :return: output tensor
        """
        X_shortcut = X
        # branch 1
        branch_1 = Conv2D_BN(X, filters=192, kernel_size=1, strides=1, padding="same",
                             name="Resnet_B_{}_conv2d_branch_1_1".format(iteration))
        # branch 2
        branch_2 = Conv2D_BN(X, filters=128, kernel_size=1, strides=1, padding="same",
                             name="Resnet_B_{}_conv2d_branch_2_1".format(iteration))
        branch_2 = Conv2D_BN(branch_2, filters=160, kernel_size=(1, 7), strides=1, padding="same",
                             name="Resnet_B_{}_conv2d_branch_2_2".format(iteration))
        branch_2 = Conv2D_BN(branch_2, filters=192, kernel_size=(7, 1), strides=1, padding="same",
                             name="Resnet_B_{}_conv2d_branch_2_3".format(iteration))
        branches = [branch_1, branch_2]
        X = Concatenate(axis=CHANNEL_AXIS, name="Resnet_B_{}_concatenate".format(iteration))(branches)
        X = Conv2D_BN(X, filters=1152, kernel_size=1, strides=1, padding="same",
                      name="Resnet_B_{}_conv2d_final".format(iteration), activation=None, use_bn=False)
        X = shortcut_summation(X, X_shortcut, scale, name="Resnet_B_{}_shortcut_sum".format(iteration))
        X = Activation("relu", name="Resnet_B_{}_relu".format(iteration))(X)
        return X

    def reduction_B(self, X):
        """Reduction-B block

        :param X: input tensor
        :return: output tensor
        """
        # branch 1
        branch_1 = MaxPool2D(pool_size=3, strides=2, padding="valid", name="Reduction_B_pool_branch_1")(X)
        # branch 2
        branch_2 = Conv2D_BN(X, filters=256, kernel_size=1, strides=1, padding="same",
                             name="Reduction_B_conv2d_branch_2_1")
        branch_2 = Conv2D_BN(branch_2, filters=384, kernel_size=3, strides=2, padding="valid",
                             name="Reduction_B_conv2d_branch_2_2")
        # branch 3
        branch_3 = Conv2D_BN(X, filters=256, kernel_size=1, strides=1, padding="same",
                             name="Reduction_B_conv2d_branch_3_1")
        branch_3 = Conv2D_BN(branch_3, filters=288, kernel_size=3, strides=2, padding="valid",
                             name="Reduction_B_conv2d_branch_3_2")
        # branch 4
        branch_4 = Conv2D_BN(X, filters=256, kernel_size=1, strides=1, padding="same",
                             name="Reduction_B_conv2d_branch_4_1")
        branch_4 = Conv2D_BN(branch_4, filters=288, kernel_size=3, strides=1, padding="same",
                             name="Reduction_B_conv2d_branch_4_2")
        branch_4 = Conv2D_BN(branch_4, filters=320, kernel_size=3, strides=2, padding="valid",
                             name="Reduction_B_conv2d_branch_4_3")
        branches = [branch_1, branch_2, branch_3, branch_4]
        X = Concatenate(axis=CHANNEL_AXIS, name="Reduction_B_concatenate")(branches)
        return X

    def resnet_C(self, X, scale=0.2, iteration=None):
        """Third Inception-Resnet block Inception-ResNet-C, 8x8 (given the default input size is used)

        :param X: input tensor
        :param scale: scaling factor for output tensor
        :param iteration: iteration index (required for layer name generation)
        :return: output tensor
        """
        X_shortcut = X
        # branch 1
        branch_1 = Conv2D_BN(X, filters=192, kernel_size=1, strides=1, padding="same",
                             name="Resnet_C_{}_conv2d_branch_1_1".format(iteration))
        # branch 2
        branch_2 = Conv2D_BN(X, filters=192, kernel_size=1, strides=1, padding="same",
                             name="Resnet_C_{}_conv2d_branch_2_1".format(iteration))
        branch_2 = Conv2D_BN(branch_2, filters=224, kernel_size=(1, 3), strides=1, padding="same",
                             name="Resnet_C_{}_conv2d_branch_2_2".format(iteration))
        branch_2 = Conv2D_BN(branch_2, filters=256, kernel_size=(3, 1), strides=1, padding="same",
                             name="Resnet_C_{}_conv2d_branch_2_3".format(iteration))
        branches = [branch_1, branch_2]
        X = Concatenate(axis=CHANNEL_AXIS, name="Resnet_C_{}_concatenate".format(iteration))(branches)
        X = Conv2D_BN(X, filters=2144, kernel_size=1, strides=1, padding="same",
                      name="Resnet_C_{}_conv2d_final".format(iteration))
        X = shortcut_summation(X, X_shortcut, scale, name="Resnet_C_{}_shortcut_sum".format(iteration))
        X = Activation("relu", name="Resnet_C_{}_relu".format(iteration))(X)
        return X

    def compile(self, loss=contrastive_loss, optimizer="adam", metrics=("accuracy"), plot=True, plot_path="model.png"):
        """Model compilation

        :param loss: loss function
        :param optimizer: optimizer
        :param metrics: list of metrics to track
        :param plot: whether to save the architecture plot to png
        :param plot_path: file for model plot saving
        :return: compiled model
        """
        X = Input(shape=self.input_shape)
        X_out = self.stem(X)
        for i in range(5):
            X_out = self.resnet_A(X_out, iteration=i)
        X_out = self.reduction_A(X_out)
        for i in range(10):
            X_out = self.resnet_B(X_out, iteration=i)
        X_out = self.reduction_B(X_out)
        for i in range(5):
            X_out = self.resnet_C(X_out, iteration=i)
        X_out = GlobalAveragePooling2D(data_format=K.image_data_format(), name="GlobalAveragePooling")(X_out)
        X_out = Dropout(rate=1 - self.dropout_keep_prob, name="Dropout")(X_out)  # rate = 1 - keep_prob
        X_out = Dense(self.embedding_shape, name="Dense")(X_out)
        model = Model(inputs=X, outputs=X_out)
        if plot:
            self.plot_model(model, plot_path)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def plot_model(self, model, plot_path):
        plot_model(model, show_shapes=True, to_file="model.png")