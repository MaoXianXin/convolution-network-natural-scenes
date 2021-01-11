import os
import pandas as pd

flower_names = os.listdir('/home/mao/Downloads/datatset/flowers')

flower_img_paths = []
flower_label_paths = []
for flower_name in flower_names:
    flowerNames = os.listdir('/home/mao/Downloads/datatset/flowers/' + flower_name)
    for img_name in flowerNames:
        flower_img_paths.append(os.path.join('gs://flower_photos_5/'+flower_name, img_name))
        flower_label_paths.append(flower_name)


flower_gcs_dataframe = pd.DataFrame(flower_img_paths)
flower_gcs_dataframe['1'] = flower_label_paths
flower_gcs_dataframe.to_csv('flowers.csv', index=False, index_label=False)