from collections import namedtuple
import numpy as np
import tensorflow as tf
import six
from tensorflow.python.training import moving_averages



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',

           'resnet152']
model_urls = {

  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,activate_before_residual=False):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
    # 第1子层
    with tf.variable_scope('sub1'):
      # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)
    # 第2子层
    with tf.variable_scope('sub2'):
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 3x3卷积，步长为1，通道数不变(out_filter/4)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])
    # 第3子层
    with tf.variable_scope('sub3'):
      # BN和ReLU激活
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])
    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
      if in_filter != out_filter:
        # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      # 合并残差
      x += orig_x
    tf.logging.info('image after unit %s', x.get_shape())

    return x
    
 
def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):

    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        # 获取残差直连
        orig_x = x
       # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    # 第1子层
    with tf.variable_scope('sub1'):
      # 3x3卷积，使用输入步长，通道数(in_filter -> out_filter)
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 3x3卷积，步长为1，通道数不变(out_filter)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
      if in_filter != out_filter:
        # 均值池化，无补零

        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        # 通道补零(第4维前后对称补零)
        orig_x = tf.pad(orig_x, 
                        [[0, 0], 
                         [0, 0], 
                         [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]
                        ])
      # 合并残差

      x += orig_x
    tf.logging.debug('image after unit %s', x.get_shape())
    return x
    
    


def _build_model(self):
    with tf.variable_scope('init'):
      x = self._images
      """第一层卷积（3,3x3/1,16）"""
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
    # 残差网络参数
    strides = [1, 2, 2]
# 激活前置
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
   # bottleneck残差单元模块

      res_func = self._bottleneck_residual

    # 通道数量
      filters = [16, 64, 128, 256]
    else:
      # 标准残差单元模块
      res_func = self._residual
      # 通道数量
      filters = [16, 16, 32, 64]
    # 第一组

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], 
                   self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    # 第二组



    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], 
                   self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, self.hps.num_residual_units):

with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    # 第三组

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
        
     #池化层，得到globle feature和local feature 
    return x
    
    
    
    
    
  # 把步长值转换成tf.nn.conv2d需要的步长数组
  def _stride_arr(self, stride):    
    return [1, stride, stride, 1] 
    
    
    
  # Batch Normalization批归一化
  # ((x-mean)/var)*gamma+beta
  def _batch_norm(self, name, x):
    with tf.variable_scope(name):
      # 输入通道维数
      params_shape = [x.get_shape()[-1]]
      # offset
      beta = tf.get_variable('beta', 
                             params_shape, 
                             tf.float32,
                             initializer=tf.constant_initializer(0.0, tf.float32))
      # scale
      gamma = tf.get_variable('gamma', 
                              params_shape, 
                              tf.float32,
                              initializer=tf.constant_initializer(1.0, tf.float32))
      if self.mode == 'train':
        # 为每个通道计算均值、标准差
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        # 新建或建立测试阶段使用的batch均值、标准差
        moving_mean = tf.get_variable('moving_mean', 
                                      params_shape, tf.float32,
                                      initializer=tf.constant_initializer(0.0, tf.float32),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', 
                                          params_shape, tf.float32,
                                          initializer=tf.constant_initializer(1.0, tf.float32),
                                          trainable=False)
        # 添加batch均值和标准差的更新操作(滑动平均)
        # moving_mean = moving_mean * decay + mean * (1 - decay)
        # moving_variance = moving_variance * decay + variance * (1 - decay)
        self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                        moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                        moving_variance, variance, 0.9))



      else:
        # 获取训练中积累的batch均值、标准差
        mean = tf.get_variable('moving_mean', 
                               params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32),
                               trainable=False)
        variance = tf.get_variable('moving_variance', 
                                   params_shape, tf.float32,
                                   initializer=tf.constant_initializer(1.0, tf.float32),
                                   trainable=False)

        # 添加到直方图总结
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)

      # BN层：((x-mean)/var)*gamma+beta
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y
      
      
      
      
 

  # 2D卷积

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      # 获取或新建卷积核，正态随机初始化
      kernel = tf.get_variable(
              'DW', 
              [filter_size, filter_size, in_filters, out_filters],
              tf.float32, 
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      # 计算卷积
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
  # leaky ReLU激活函数，泄漏参数leakiness为0就是标准ReLU
  
  

  def _relu(self, x, leakiness=0.0):

    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')



  
