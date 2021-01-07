from tensorflow.keras.models import load_model
from utils.data_generator import test_generator, pred_generator
from utils.image_plot import plot_images
import json
import requests
import numpy as np

test_gen = test_generator(
    data_dir='../dataset/natural-scenes/seg_test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

pred_gen = pred_generator(
    data_dir='../dataset/natural-scenes/seg_pred',
    target_size=(224, 224),
    batch_size=32,
    class_mode=None)

'''
模型载入tf.keras.models.load_model
用到的参数：
- filepath：载入模型存放的路径。

模型验证tf.keras.Sequential.evaluate
用到的参数：
- x：输入的验证集，可以用ImageDataGenerator读取的数据。

模型预测tf.keras.Sequential.predict
用到的参数：
- x：需要做预测的数据集，可以用ImageDataGenerator读取的数据。
'''

model_path = './models/natural_scenes/1'
loaded_model = load_model(filepath=model_path)
# loss, accuracy = loaded_model.evaluate(x=test_gen)

pred_batch = pred_gen.next()
pred_result = loaded_model.predict(x=pred_batch)
plot_images(pred_batch, pred_result)

data = json.dumps({"signature_name": "serving_default", "instances": pred_batch.tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/scenes_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
# print(predictions)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# print(np.argmax(predictions[0]))
# print(pred_result[0])
print('The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[np.argmax(pred_result[0])], np.argmax(pred_result[0])))