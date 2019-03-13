import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = "/home/wyh/3dunet/model_rand/model_30epoch.ckpt"
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')