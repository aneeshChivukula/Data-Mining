import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")
tf.app.flags.DEFINE_integer('height', 462,
                            """image height""")
tf.app.flags.DEFINE_integer('width', 581,
                            """image width""")
tf.app.flags.DEFINE_integer('depth', 3,
                            """image width""")

class Parser():
    def parse_example_proto(self,example_serialized):
      """Parses an Example proto containing a training example of an image.
      The output of the build_image_data.py image preprocessing script is a dataset
      containing serialized Example protocol buffers. Each Example proto contains
      the following fields:
        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>
      Args:
        example_serialized: scalar Tensor tf.string containing a serialized
          Example protocol buffer.
      Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged as
          [ymin, xmin, ymax, xmax].
        text: Tensor tf.string containing the human-readable label.
      """
      # Dense features in Example proto.
      feature_map = {
          'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
          'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                  default_value=-1),
          'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                 default_value=''),
      }
      sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
      # Sparse features in Example proto.
      feature_map.update(
          {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                       'image/object/bbox/ymin',
                                       'image/object/bbox/xmax',
                                       'image/object/bbox/ymax']})
      feature_map.update({k: sparse_float32 for k in [
                                                      'image/height',
                                                      'image/width']})
      features = tf.parse_single_example(example_serialized, feature_map)
#       print('features',features)
      
      label = tf.cast(features['image/class/label'], dtype=tf.int32)
#       print('label',label)
    
#       xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
#       ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
#       xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
#       ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
#       print('xmin',xmin)
#       print('ymin',ymin)
#       print('xmax',xmax)
#       print('ymax',ymax)
#       print('features[image/encoded]',features['image/encoded'])
#       print('features[image/height]',features['image/height'])
#       print('features[image/width]',features['image/width'])

      # Note that we impose an ordering of (y, x) just to make life difficult.
#       bbox = tf.concat(0, [ymin, xmin, ymax, xmax])
    
      # Force the variable number of bounding boxes into the shape
      # [1, num_boxes, coords].
#       bbox = tf.expand_dims(bbox, 0)
#       bbox = tf.transpose(bbox, [0, 2, 1])
    
      return features['image/encoded'], label
#       return features['image/encoded'], label, bbox, features['image/class/text']


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
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.reshape(image, shape=[FLAGS.height, FLAGS.width, FLAGS.depth])
        print 'image_buffer',image_buffer
        print 'image',image
        return image

#     def image_preprocessing(self,image_buffer, bbox):
    def image_preprocessing(self,image_buffer):
          image = self.decode_jpeg(image_buffer)
          return image
        # distort_image, eval_image give various bounding boxes image/math operations in tf
        # by resizing images we can have squared images instead of rectangular images
        # rectangular images are the defaults in tensorflow encoding TF format
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
        # https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
        # http://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow