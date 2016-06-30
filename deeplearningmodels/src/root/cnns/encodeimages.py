import os
import sys
import tensorflow as tf
import numpy as np
import threading

train_dir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Train'
test_dir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/101_ObjectCategories_Test/'
labels_file = '/home/aneesh/models-master/inception/labels.txt'
output_dir = '/home/aneesh/Documents/AdversarialLearningDatasets/Caltech101/SerializedObjectCategories/'

train_shards = 10
test_shards=24
imagespershard=10

num_shards = train_shards
num_threads = 5
name = 'train'
colorspace = 'RGB'
channels = 3
image_format = 'JPEG'

unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]
print(unique_labels)

filenames = []
labels = []
humans = []

label_index = 0





class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
    
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        
    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
        return image


def process_image_files_batch(coder, name, thread_index,ranges,filenames,humans,labels,num_shards):
    num_threads = len(ranges)
    
    num_shards_per_batch = int(num_shards / num_threads)
    shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)

    
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    print num_files_in_thread
    print num_shards_per_batch
    
    counter = 0
    
    for s in xrange(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        
        output_file = os.path.join(output_dir, output_filename)
        
        writer = tf.python_io.TFRecordWriter(output_file)
        
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        
        print files_in_shard
        
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            human = humans[i]
            
            print 'label',label
            print 'label',type(label)
            
            image_data = tf.gfile.FastGFile(filename, 'r').read()
            image_buffer = coder.decode_jpeg(image_data)
#             height = image_buffer.shape[0]
#             width = image_buffer.shape[1]
            print 'image_data',image_data
            print 'image_buffer',image_buffer
            
            
            example = tf.train.Example(features=tf.train.Features(
                feature={
                'image': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_data])),
#                 bytes_list=tf.train.BytesList(value=[image_buffer.tostring()])),
                'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label])),
                'text': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[human])),
                'filename': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[os.path.basename(filename)]))
            }))

            writer.write(example.SerializeToString())
            
            shard_counter += 1
            counter += 1
            
            if not counter % 1000:
                print('[thread %d]: Processed %d of %d images in thread batch.' %
              (thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
        print('[thread %d]: Wrote %d images to %s' %
          (thread_index, shard_counter, output_file))
            
    print('[thread %d]: Wrote %d images to %d shards.' %
        (thread_index, counter, num_files_in_thread))    
    sys.stdout.flush()
    

for synset in unique_labels:
    jpeg_file_path = '%s/%s/*.jpg' % (train_dir,synset)
    matching_files = tf.gfile.Glob(jpeg_file_path)
    labels.extend([label_index] * len(matching_files))
    humans.extend([synset] * len(matching_files))
    filenames.extend(matching_files)
    label_index += 1

coder = ImageCoder()

ranges = []
threads = []

spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
print spacing

for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

print ranges
print num_shards
print num_threads


coord = tf.train.Coordinator()

for thread_index in xrange(len(ranges)):
    args = (coder,name, thread_index,ranges,filenames,humans,labels,num_shards)
#     process_image_files_batch(name, thread_index,ranges,filenames,humans,labels,num_shards)
    

    t = threading.Thread(target=process_image_files_batch, args=args)
    t.start()
    threads.append(t)

coord.join(threads)

print('Finished writing all %d images in data set.' %(len(filenames)))


sys.stdout.flush()


'''
# Complete encoding and decoding images without depending on inception parser
    # Continue with code in dataset.py and build_imagenet_data.py
    # Start from _process_dataset in build_imagenet_data.py
    # Complete encoding/decoding + modelling code loading + modelling code
# Initialize functions for multithreading code. Check coder object methods with place holders and functions before execution.
'''
