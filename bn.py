import tensorflow as tf
import numpy as np
from tensorflow.keras import initializers
import tensorflow.keras.backend as K


def sqrt_init(shape, dtype=None,partition_info=None):
    value = (1 / np.sqrt(2)) * K.ones(shape,dtype=dtype)
    return value


def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return initializers.get(init)

        
def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return initializers.serialize(init)


#此函数实现了=逆矩阵乘中心化后的数据
def complex_standardization(input_centred, Vrr, Vii, Vri,axis=-1):
    
    ndim = K.ndim(input_centred)         #整数的形式返回张量是几维的
    input_dim = K.shape(input_centred)[axis] // 2 #k.shape返回张量形状列表，这一层的输出的一半是下一层的输入,取通道数的一半，因为一半实数一半虚数
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim #variances_broadcast=[1,1,1,通道数一半]

    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >= 0 because SPD
    tau = Vrr + Vii                                   #这些Vrr的维度和输入张量除去第一个维度。
    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (Vrr * Vii) - (Vri ** 2)

    s = tf.sqrt(delta) # Determinant of square root matrix
    t = tf.sqrt(tau + 2 * s)

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = K.reshape(Wrr, variances_broadcast)
    broadcast_Wri = K.reshape(Wri, variances_broadcast)
    broadcast_Wii = K.reshape(Wii, variances_broadcast)

    cat_W_4_real = K.concatenate([broadcast_Wrr, broadcast_Wii], axis=axis)
    cat_W_4_imag = K.concatenate([broadcast_Wri, broadcast_Wri], axis=axis)

    if ndim == 2:
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif ndim==4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
            'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = K.concatenate([centred_imag, centred_real], axis=axis)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output

#此函数根据gamma和beta计算BN的结果
def ComplexBN(input_centred, Vrr, Vii, Vri, beta,gamma_rr, gamma_ri, gamma_ii, scale=True,center=True, axis=-1):

    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 2

    if scale:
        standardized_output = complex_standardization(
            input_centred, Vrr, Vii, Vri,
            axis=axis
        )

        # Now we perform th scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed + gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed + gamma_ii * x_imag_normed + beta_imag
        
        broadcast_gamma_rr = K.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = K.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_ii = K.reshape(gamma_ii, gamma_broadcast_shape)

        cat_gamma_4_real = K.concatenate([broadcast_gamma_rr, broadcast_gamma_ii], axis=axis)
        cat_gamma_4_imag = K.concatenate([broadcast_gamma_ri, broadcast_gamma_ri], axis=axis)
        if ndim==2:
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim==4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output = K.concatenate([centred_imag, centred_real], axis=axis)
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred


class ComplexBatchNormalization(object):
    def __init__(self,input_R,input_I,
                 axis=-1,
                 is_training = True,
                 momentum = 0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros'):

        self.supports_masking = True
        self.is_training = is_training
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer              = sanitizedInitGet(beta_initializer) #sqrt_init的就用根号初始化
        self.gamma_diag_initializer        = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer         = sanitizedInitGet(gamma_off_initializer)

        input_x = K.concatenate([input_R,input_I],axis = self.axis)
        self.build(input_x.get_shape().as_list())
        self.out = self.call(input_x,is_training = self.is_training)

    def build(self, input_shape):  #构建可学习的参数

        ndim = len(input_shape)

        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        param_shape = (input_shape[self.axis] // 2,)

        if self.scale:
            self.gamma_rr = tf.get_variable(shape=param_shape,
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            dtype = tf.float32)
            self.gamma_ii = tf.get_variable(shape=param_shape,
                                            name='gamma_ii',
                                            initializer=self.gamma_diag_initializer,
                                            dtype = tf.float32)
            self.gamma_ri = tf.get_variable(shape=param_shape,
                                            name='gamma_ri',
                                            initializer=self.gamma_off_initializer,
                                            dtype = tf.float32)

        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None


        if self.center:
            self.beta = tf.get_variable(shape=(input_shape[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        dtype = tf.float32)
        else:
            self.beta = None


    def call(self, inputs, is_training=None): #计算input_ceter 和 vrr、vii等
        input_shape = inputs.get_shape().as_list()
        ndim = len(input_shape)
        reduction_axes = list(range(ndim-1))#【1，2，3，4】
        input_dim = input_shape[self.axis] // 2
        mu,_ = tf.nn.moments(inputs,reduction_axes,name='moments')
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)#把mu搞到第四维度上.

        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
        centred_squared = input_centred ** 2
        if ndim==2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim==4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = K.mean(
                centred_squared_real,
                axis=reduction_axes
            ) + self.epsilon
            Vii = K.mean(
                centred_squared_imag,
                axis=reduction_axes
            ) + self.epsilon
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = K.mean(
                centred_real * centred_imag,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        ema = tf.train.ExponentialMovingAverage(self.momentum)

        def mean_var_with_update():
            ema_apply_op = ema.apply([mu,Vrr,Vii,Vri])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mu),tf.identity(Vrr),tf.identity(Vii),tf.identity(Vri),

        mean,VRR,VII,VRI = tf.cond(tf.equal(is_training, True), mean_var_with_update, lambda: (
                ema.average(mu), ema.average(Vrr),ema.average(Vii),ema.average(Vri)))

        input_bn = ComplexBN(
            inputs - K.reshape(mean, broadcast_mu_shape), VRR, VII, VRI,
            self.beta, self.gamma_rr, self.gamma_ri,
            self.gamma_ii, self.scale, self.center,
            axis=self.axis)
        if ndim==2:
            return input_bn[:, :input_dim],input_bn[ :, input_dim:]
        else:
            return input_bn[:, :, :, :input_dim],input_bn[:, :, :, input_dim:]