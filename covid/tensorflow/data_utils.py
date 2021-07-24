import tensorflow as tf 

def get_decode_fn(img_extension, channels):
    def decode_fn(path, label): 
        file_bytes = tf.io.read_file(path)
        if img_extension == 'png':
            img = tf.io.decode_image(file_bytes, channels=channels, expand_animations = False)
        elif img_extension in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=channels)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    return decode_fn