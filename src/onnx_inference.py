import os
from utils.data_generator import train_val_generator
import numpy as np
import onnxruntime as ort
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_gen = train_val_generator(
    data_dir='../dataset/natural-scenes/seg_train',
    target_size=(224, 224),    # 把图片的h和w从64变成150，增大图片的分辨率
    batch_size=32,
    class_mode='categorical',
    subset='training')

val_gen = train_val_generator(
    data_dir='../dataset/natural-scenes/seg_train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# ImageDataGenerator的返回结果是个迭代器，调用一次才会吐一次结果，可以使用.next()函数分批读取图片。
# 取15张训练集图片进行查看
train_batch, train_label_batch = train_gen.next()
# plot_images(train_batch, train_label_batch)

# 取15张测试集图片进行查看
val_batch, val_label_batch = val_gen.next()
# plot_images(val_batch, val_label_batch)

sess_ort = ort.InferenceSession('./natural.onnx')
res = sess_ort.run(output_names=['Identity:0'], input_feed={'onnx_input:0':np.array(train_batch)})
print(np.array(res).squeeze().shape)
print(np.argmax(np.array(res).squeeze(), axis=1))
print(np.argmax(np.array(train_label_batch), axis=1))