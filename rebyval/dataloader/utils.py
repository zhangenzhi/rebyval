import fnmatch, os
import re
import tensorflow as tf
import tarfile
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


def convert_imagenet_to_tfrecords(input_dirs, output_dirs):
    # generate label from meta data
    metadata = scipy_io.loadmat('./examples/dataset/imagenet/meta.mat')
    import pdb
    pdb.set_trace()
    synsets_info = metadata['synsets']
    feature_dict = {}
    for item in synsets_info:
        set_info = item[0]
        feature_dict[set_info[1][0]] = set_info[0][0][0]

    # open imge.jpeg and save as tfrecord by 5000 a group
    synsets_file = os.listdir(input_dirs)
    image_strings_buffer = []
    for synset in synsets_file:
        synset_path = os.path.join(input_dirs, synset)
        image_jpeg = os.listdir(synset_path)

        for img in image_jpeg:
            img_path = os.path.join(synset_path, img)
            image_strings_buffer.append((open(img_path, 'rb').read(), feature_dict[synset]))

            if len(image_strings_buffer) == 5000:
                num_tfrecords = len(os.listdir(output_dirs))
                record_file = "{}.tfrecords".format(num_tfrecords)
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
        with tf.io.TFRecordWriter(record_file) as writer:
            for image_string, label in image_strings_buffer:
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
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=[value]))


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
