import tensorflow as tf

# HYPER PARAMETERS
IMAGE_SIZE = 512
SHUFFLE_BUFFER_SIZE = 512
# AUTOTUNE = tf.data.experimental.AUTOTUNE ## The Capacity for async 
# strategy = tf.distribute.get_strategy() ##
# BATCH_SIZE = 2 * strategy.num_replicas_in_sync
BATCH_SIZE = 2
AUTOTUNE = 3

def decode_image(image_raw):
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def parse_tfrecord(tfrecord):
    features = {'image_id': tf.io.FixedLenFeature([], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}

    parsed_image_dataset = tf.io.parse_single_example(tfrecord, features)
    image_id = tf.cast(parsed_image_dataset['image_id'], tf.int32)
    image = decode_image(parsed_image_dataset['image_raw'])
    label = tf.cast(parsed_image_dataset['label'], tf.int32)
    return image_id, image, label

def load_dataset(filenames):
    raw_image_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    image_dataset = raw_image_dataset.map(parse_tfrecord, num_parallel_calls=AUTOTUNE) 
    
    return image_dataset

def check_format(image_id, image, label):
    return image_id


def arcface_format(image_id, image, label):
    return {'input/image': image, 'input/label': label}, label


def arcface_evaluation_format(image_id, image, label):
    return image

def arcface_test_format(image_id, image, label):
    return image_id, image


def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    return image


def get_check_dataset(filenames):
    ds = load_dataset(filenames)
    ds = ds.map(check_format, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

    
def get_training_dataset(filenames):
    ds = load_dataset(filenames)
    ds = ds.map(lambda image_id, image, label: (image, label))
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda image, label: (augment(image), label), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda image, label: ({'input/image': image, 'input/label': label}, label))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds
    
    
def get_validation_dataset(filenames):
    ds = load_dataset(filenames)
    ds = ds.map(arcface_format, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds
    

def get_evaluation_dataset(filenames):
    ds = load_dataset(filenames)
    ds = ds.map(arcface_evaluation_format, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


def get_test_dataset(filenames):
    ds = load_dataset(filenames)
    ds = ds.map(arcface_evaluation_format, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds
