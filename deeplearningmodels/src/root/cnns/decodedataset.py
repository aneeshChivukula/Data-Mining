import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('height', 100,
                            """image height""")
tf.app.flags.DEFINE_integer('width', 300,
                            """image width""")
tf.app.flags.DEFINE_integer('depth', 3,
                            """image width""")

class Parser():
    def parse_example_proto(self,example_serialized):
        feature_map = {
          'image': tf.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
          'label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                  default_value=-1),
          'text': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value=''),
         'filename': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value='')
        }
        print('example_serialized',example_serialized)
        features = tf.parse_single_example(example_serialized, feature_map)
#         label = tf.cast(features['image/class/label'], dtype=tf.int32)
    
        return features['image'], features['label']


    def decode_jpeg(self,image_buffer, scope=None):
      """Decode a JPEG string into one 3-D float image Tensor.
      Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for op_scope.
      Returns:
        3-D float Tensor with values ranging from [0, 1).
      """
      with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
        image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.depth)
        
#         print(image,image.__class__)
        
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reshape(image, shape=[FLAGS.height, FLAGS.width, FLAGS.depth])
        return image

#     def image_preprocessing(self,image_buffer, bbox):
    def image_preprocessing(self,image_buffer):
          return self.decode_jpeg(image_buffer)




        
"""     
Check build_imagenet_data.py and image_processing.py for encoding and decoding logic in inception model
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
        # https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
        # http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow
"""