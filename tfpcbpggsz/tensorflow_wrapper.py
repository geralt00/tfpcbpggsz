import os
import warnings
# default configurations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # for Mac
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# pylint: disable=no-member
try:
    tf_version = int(str(tf.__version__).split(".")[0])
except Exception:
    tf_version = 2

def set_gpu_mem_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        except ValueError as e:
            print(e)


if "TF_PWA_GPU_FULL_MEM" in os.environ:
    if os.environ["TF_PWA_GPU_FULL_MEM"] == "0":
        set_gpu_mem_growth()
else:
    set_gpu_mem_growth()


class Module(object):
    pass


tensorflow_wrapper = Module()


def regist_function(name, var=None, base_mod=tensorflow_wrapper):
    mod = base_mod
    names = name.split(".")
    for i in names[:-1]:
        if not hasattr(mod, i):
            setattr(mod, i, Module())
        mod = getattr(mod, i)

    def wrapper(f):
        if hasattr(mod, names[-1]):
            warnings.warn("{} already exists.".format(name))
        setattr(mod, names[-1], f)
        return f

    if var is None:
        return wrapper
    else:
        return wrapper(var)


# @regist_function("cross", base_mod=tf)
def numpy_cross(a, b):
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    a_0 = tf.zeros_like(b)
    b_0 = tf.zeros_like(a)
    a = a + a_0
    b = b + b_0
    # shape = tf.broadcast_static_shape(a.shape, b.shape)
    # a = tf.broadcast_to(a, shape)
    # b = tf.broadcast_to(b, shape)
    ret = tf.linalg.cross(a, b)
    return ret


# regist_function("sum", tf.reduce_sum, base_mod=tf)
regist_function("arctan2", tf.math.atan2, base_mod=tf)


# from .jax_wrapper import tf
