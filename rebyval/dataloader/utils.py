import re
import tarfile
import random
import fnmatch, os
import tensorflow as tf
import numpy as np
from tensorflow.io import gfile
from scipy import io as scipy_io


def glob_tfrecords(input_dirs, glob_pattern="example", recursively=False):
    file_path_list = []
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    for root_path in input_dirs:
        assert gfile.exists(root_path), "{} does not exist.".format(root_path)
        if not gfile.isdir(root_path):
            file_path_list.append(root_path)
            continue
        if not recursively:
            for filename in gfile.listdir(root_path):
                if fnmatch.fnmatch(filename, glob_pattern):
                    file_path_list.append(os.path.join(root_path, filename))
        else:
            for dir_path, _, filename_list in gfile.walk(root_path):
                for filename in filename_list:
                    if fnmatch.fnmatch(filename, glob_pattern):
                        file_path_list.append(os.path.join(dir_path, filename))
    return file_path_list

def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    
    # subtract mean
    # x_train = np.reshape(train_images, (train_images.shape[0], -1)).astype('float32')
    # x_test = np.reshape(test_images, (test_images.shape[0], -1)).astype('float32')
    # mean_image = np.mean(x_train, axis=0).astype('uint8')
    # train_images = train_images - mean
    # test_images = test_images - mean
    
    return train_images, test_images

def unpack_tarfile(input_dirs):
    tarfiles = os.listdir(input_dirs)

    for f in tarfiles:
        tarfile_path = os.path.join(input_dirs, f)

        file_pattern = re.findall(r"\d+", f)[0]
        file_save_path = os.path.join(input_dirs, file_pattern)
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)

        with tarfile.open(tarfile_path) as tar:
            tar.extractall(path=file_save_path)

        os.remove(tarfile_path)


def convert_imagenet_validset_to_tfrecords(input_dirs, output_dirs, config_path=None):
    # validation filepath
    valid_labels = {}
    filename = config_path if config_path else './examples/dataset/imagenet/ILSVRC2012_validation_ground_truth.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        prefix = "ILSVRC2012_val_"
        for i in range(len(lines)):
            valid_labels[prefix + '{:08d}.JPEG'.format(i + 1)] = int(lines[i])

    # open image.jpeg and save as tfrecord by 5000 a group
    image_jpegs = os.listdir(input_dirs)
    image_strings_buffer = []

    for img in image_jpegs:
        img_path = os.path.join(input_dirs, img)
        image_strings_buffer.append((open(img_path, 'rb').read(), valid_labels[img]-1))

        if len(image_strings_buffer) == 5000:
            num_tfrecords = len(os.listdir(output_dirs))
            record_file = "{}.tfrecords".format(num_tfrecords)
            record_file = os.path.join(output_dirs, record_file)
            with tf.io.TFRecordWriter(record_file) as writer:
                for image_string, label in image_strings_buffer:
                    tf_example = _image_example(image_string=image_string, label=label)
                    writer.write(tf_example.SerializeToString())
            print("{} convert finished.".format(record_file))
            image_strings_buffer = []

    # last tfrecord
    if len(image_strings_buffer) != 0:
        num_tfrecords = len(os.listdir(output_dirs))
        record_file = "{}.tfrecords".format(num_tfrecords)
        record_file = os.path.join(output_dirs, record_file)
        with tf.io.TFRecordWriter(record_file) as writer:
            for image_string, label in image_strings_buffer:
                tf_example = _image_example(image_string=image_string, label=label)
                writer.write(tf_example.SerializeToString())
        print("{} is the last and convert finished.".format(record_file))


def convert_imagenet_trainset_to_tfrecords(input_dirs, output_dirs):
    # generate label from meta data
    metadata = scipy_io.loadmat('./examples/dataset/imagenet/meta.mat')
    synsets_info = metadata['synsets']
    feature_dict = {}
    for item in synsets_info:
        set_info = item[0]
        feature_dict[set_info[1][0]] = set_info[0][0][0]

    # pre-shuffle images in image path
    synsets_file = os.listdir(input_dirs)
    image_strings_buffer = []
    for synset in synsets_file:
        synset_path = os.path.join(input_dirs, synset)
        image_jpeg = os.listdir(synset_path)
        for img in image_jpeg:
            img_path = os.path.join(synset_path, img)
            image_strings_buffer.append((img_path, feature_dict[synset]-1))
    random.shuffle(image_strings_buffer)

    # open image.jpeg as string and save into tfrecord by 5000 samples a group
    tmp_buffer = []
    for img_path, label in image_strings_buffer:
        tmp_buffer.append((open(img_path, mode='rb').read(), label))
        if len(tmp_buffer) == 5000:
            num_tfrecords = len(os.listdir(output_dirs))
            record_file = os.path.join(output_dirs, "{}.tfrecords".format(num_tfrecords))
            with tf.io.TFRecordWriter(record_file) as writer:
                for image_string, label in tmp_buffer:
                    tf_example = _image_example(image_string=image_string, label=label)
                    writer.write(tf_example.SerializeToString())
            print("{} convert finished.".format(record_file))
            tmp_buffer = []

    # last tfrecord
    if len(tmp_buffer) != 0:
        num_tfrecords = len(os.listdir(output_dirs))
        record_file = os.path.join(output_dirs, "{}.tfrecords".format(num_tfrecords))
        with tf.io.TFRecordWriter(record_file) as writer:
            for image_string, label in tmp_buffer:
                tf_example = _image_example(image_string=image_string, label=label)
                writer.write(tf_example.SerializeToString())
        print("{} is the last and convert finished.".format(record_file))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == '__main__':
    input_dirs = "/home/work/dataset/ILSVRC2012/downloads/manual/valid"
    output_dirs = "/home/work/dataset/ILSVRC2012/downloads/manual/valid_records"
    metadata = convert_imagenet_validset_to_tfrecords(input_dirs, output_dirs)
