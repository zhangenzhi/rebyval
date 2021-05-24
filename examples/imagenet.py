import io
import os
import tarfile
import logging

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

dataset_name = "imagenet2012"
manual_dataset_dir = "/home/work/dataset/ILSVRC2012"

_LABELS_FNAME = './imagenet/imagenet2012_labels.txt'

# This file contains the validation labels, in the alphabetic order of
# corresponding image names (and not in the order they have been added to the
# tar file).
_VALIDATION_LABELS_FNAME = './imagenet/imagenet2012_validation_labels.txt'

# From https://github.com/cytsai/ilsvrc-cmyk-image-list
CMYK_IMAGES = [
    'n01739381_1309.JPEG',
    'n02077923_14822.JPEG',
    'n02447366_23489.JPEG',
    'n02492035_15739.JPEG',
    'n02747177_10752.JPEG',
    'n03018349_4028.JPEG',
    'n03062245_4620.JPEG',
    'n03347037_9675.JPEG',
    'n03467068_12171.JPEG',
    'n03529860_11437.JPEG',
    'n03544143_17228.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n03961711_5286.JPEG',
    'n04033995_2932.JPEG',
    'n04258138_17003.JPEG',
    'n04264628_27969.JPEG',
    'n04336792_7448.JPEG',
    'n04371774_5854.JPEG',
    'n04596742_4225.JPEG',
    'n07583066_647.JPEG',
    'n13037406_4650.JPEG',
]

PNG_IMAGES = ['n02105855_2933.JPEG']


class Imagenet2012(tfds.core.GeneratorBasedBuilder):

    @staticmethod
    def _get_validation_labels(self, val_path):
        """Returns labels for validation.
        Args:
          val_path: path to TAR file containing validation images. It is used to
            retrieve the name of pictures and associate them to labels.
        Returns:
          dict, mapping from image name (str) to label (str).
        """
        labels_path = tfds.core.tfds_path(_VALIDATION_LABELS_FNAME)
        with tf.io.gfile.GFile(os.fspath(labels_path)) as labels_f:
            # `splitlines` to remove trailing `\r` in Windows
            labels = labels_f.read().strip().splitlines()
        with tf.io.gfile.GFile(val_path, 'rb') as tar_f_obj:
            tar = tarfile.open(mode='r:', fileobj=tar_f_obj)
            images = sorted(tar.getnames())
        return dict(zip(images, labels))

    def _fix_image(self, image_fname, image):
        """Fix image color system and format starting from v 3.0.0."""
        if self.version < '3.0.0':
            return image
        if image_fname in CMYK_IMAGES:
            image = io.BytesIO(tfds.core.utils.jpeg_cmyk_to_rgb(image.read()))
        elif image_fname in PNG_IMAGES:
            image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()))
        return image

    def _split_generators(self, dl_manager):
        train_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_train.tar')
        val_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_val.tar')
        test_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_val.tar')

        splits = []
        _add_split_if_exists(
            split_list=splits,
            split=tfds.Split.TRAIN,
            split_path=train_path,
            dl_manager=dl_manager,
        )

        _add_split_if_exists(
            split_list=splits,
            split=tfds.Split.VALIDATION,
            split_path=val_path,
            dl_manager=dl_manager,
            validation_labels=self._get_validation_labels(val_path),
        )

        _add_split_if_exists(
            split_list=splits,
            split=tfds.Split.TEST,
            split_path=test_path,
            dl_manager=dl_manager,
            labels_exist=False,
        )

        if not splits:
            raise AssertionError(
                'ImageNet requires manual download of the data. Please download '
                'the data and place them into:\n'
                f' * train: {train_path}\n'
                f' * test: {test_path}\n'
                f' * validation: {val_path}\n'
                'At least one of the split should be available.')
        return

    def _generate_examples(self,
                           archive,
                           validation_labels=None,
                           labels_exist=True):
        """Yields examples."""
        if not labels_exist:  # Test split
            for key, example in self._generate_examples_test(archive):
                yield key, example
        if validation_labels:  # Validation split
            for key, example in self._generate_examples_validation(
                    archive, validation_labels):
                yield key, example
        # Training split. Main archive contains archives names after a synset noun.
        # Each sub-archive contains pictures associated to that synset.
        for fname, fobj in archive:
            label = fname[:-4]  # fname is something like 'n01632458.tar'
            # TODO(b/117643231): in py3, the following lines trigger tarfile module
            # to call `fobj.seekable()`, which Gfile doesn't have. We should find an
            # alternative, as this loads ~150MB in RAM.
            fobj_mem = io.BytesIO(fobj.read())
            for image_fname, image in tfds.download.iter_archive(
                    fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
                image = self._fix_image(image_fname, image)
                record = {
                    'file_name': image_fname,
                    'image': image,
                    'label': label,
                }
                yield image_fname, record

    def _generate_examples_validation(self, archive, labels):
        for fname, fobj in archive:
            record = {
                'file_name': fname,
                'image': fobj,
                'label': labels[fname],
            }
            yield fname, record

    def _generate_examples_test(self, archive):
        for fname, fobj in archive:
            record = {
                'file_name': fname,
                'image': fobj,
                'label': -1,
            }
            yield fname, record


def _add_split_if_exists(split_list, split, split_path, dl_manager, **kwargs):
    """Add split to given list of splits only if the file exists."""
    if not tf.io.gfile.exists(split_path):
        logging.warning(
            'ImageNet 2012 Challenge %s split not found at %s. '
            'Proceeding with data generation anyways but the split will be '
            'missing from the dataset...',
            str(split),
            split_path,
        )
    else:
        split_list.append(
            tfds.core.SplitGenerator(
                name=split,
                gen_kwargs={
                    'archive': dl_manager.iter_archive(split_path),
                    **kwargs
                },
            ), )


if __name__ == '__main__':
    # imagenet_ds = ImageNet2012()
    dataset_name = "imagenet2012"
    manual_dataset_dir = "/home/work/dataset/ILSVRC2012"
    # builder = tfds.builder(dataset_name, data_dir=manual_dataset_dir)
    tfds.list_builders()
    ds = tfds.load(dataset_name, data_dir=manual_dataset_dir)
