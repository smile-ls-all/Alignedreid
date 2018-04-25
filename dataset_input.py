"""CIFAR dataset input module.
"""

import tensorflow as tf
import os
import cv2
import numpy as np

def build_input(dataset, data_path, batch_size, mode):
  """Build CIFAR image and labels.

  Args:
    dataset(数据集): Either 'cifar10' or 'cifar100'.
    data_path(数据集路径): Filename for data.
    batch_size: Input batch size.
    mode(模式）: Either 'train' or 'eval'.
  Returns:
    images(图片): Batches of images. [batch_size, image_size, image_size, 3]
    labels(类别标签): Batches of labels. [batch_size, num_classes]
  Raises:
    ValueError: when the specified dataset is not supported.
  """
  
  # 数据集参数
  image_height = 224
  image_width=224
  getTrianList()
  trans2tfRecord("train.txt","xh","file",image_height,image_width)
  image,label=read_tfRecord("xh.tfrecords")
  image_batches,label_batches = tf.train.batch([image, label], batch_size=batch_size, capacity=20)
  assert len(image_batches.get_shape()) == 4  
  assert image_batches.get_shape()[0] == 1
  assert image_batches.get_shape()[-1] == 3
  assert len(label_batches.get_shape()) == 1
  assert label_batches.get_shape()[0] == 16
  return image_batches,label_batches
  
  

def load_file(example_list_file):
    lines = np.genfromtxt(example_list_file,delimiter=" ",dtype=[('col1', 'S120'), ('col2','i8')])
    examples = []
    labels = []
    for example,label in lines:
        examples.append(example)
        labels.append(label)
    #convert to numpy array
    return np.asarray(examples),np.asarray(labels),len(lines)



def extract_image(filename,height,width):
    #print(filename)
    image = cv2.imread(filename)
    image = cv2.resize(image,(height,width))
    b,g,r = cv2.split(image)
    rgb_image = cv2.merge([r,g,b])
    return rgb_image



def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def trans2tfRecord(train_file,name,output_dir,height,width):
    if not os.path.exists(output_dir) or os.path.isfile(output_dir):
        os.makedirs(output_dir)
    _examples,_labels,examples_num = load_file(train_file)
    filename = name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i,[example,label] in enumerate(zip(_examples,_labels)):
        #print("NO{}".format(i))
        #need to convert the example(bytes) to utf-8
        #print(label)
        example = example.decode("UTF-8")
        image = extract_image(example,height,width)
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw':_bytes_feature(image_raw),
                'height':_int64_feature(image.shape[0]),
                 'width': _int64_feature(32),  
                'depth': _int64_feature(32),  
                 'label': _int64_feature(label)                        
                }))
        writer.write(example.SerializeToString())
    writer.close()


def read_tfRecord(file_tfRecord):
    queue = tf.train.string_input_producer([file_tfRecord])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
          'image_raw': tf.FixedLenFeature([], tf.string),  
          'height': tf.FixedLenFeature([], tf.int64), 
          'width':tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),  
          'label': tf.FixedLenFeature([], tf.int64)  
                    }
            )
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    #height = tf.cast(features['height'], tf.int64)
    #width = tf.cast(features['width'], tf.int64)
    image = tf.reshape(image,[224,224,3])
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_whitening(image)
    label = tf.cast(features['label'], tf.int64) 
    #print(tf.shape(image),tf.shape(features['label']))
    return image,label


def getTrianList():
    root_dir = os.getcwd()
    with open("train.txt","w") as f:
        for file in os.listdir(root_dir+'/train'):
             a=file.split('_')[0]
             f.write("train/"+file+" "+a+"\n")
