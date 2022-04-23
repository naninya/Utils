import os
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_example(img_str, file_id, label):
    feature = {
        "image_id": _int64_feature(file_id),
        "image_raw": _bytes_feature(img_str),
        "label": _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def image2tfrecord(img_dir, tfrecord_dir):
    """
    Image format to tfrecord format

    Args:
        img_dir (str): image's directory
        tfrecord_dir (str): tfrecord's directory
    """

    if not os.path.isdir(tfrecord_dir):
        os.mkdir(tfrecord_dir)
    img_file_names = os.listdir(img_dir)
    for index, img_file_name in enumerate(img_file_names):
        img_path = os.path.join(img_dir, img_file_name)
        try:
            output_path = os.path.join(tfrecord_dir, f"{os.path.splitext(img_file_name)[0]}.tfrecord")
        except Exception:
            print("Plz edit format ......")
            return
        tf_example = make_example(  
            img_str=open(img_path, 'rb').read(),
            file_id=index,
            label=index
        )

        with tf.io.TFRecordWriter(output_path) as writer:
            writer.write(tf_example.SerializeToString())