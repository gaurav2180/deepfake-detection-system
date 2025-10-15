# test_gpu.py
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Additional useful info
if tf.config.list_physical_devices('GPU'):
    print("GPU Name:", tf.config.experimental.get_device_details(tf.config.list_physical_devices('GPU')[0])['device_name'])
    print("CUDA Version:", tf.sysconfig.get_build_info()['cuda_version'])
else:
    print("No GPU detected - will use CPU only")
