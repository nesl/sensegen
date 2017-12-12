"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import tensorflow as tf


def reset_session_and_model():
    """
    Resets the TensorFlow default graph and session.
    """
    tf.reset_default_graph()
    sess = tf.get_default_session()
    if sess:
        sess.close()

        
