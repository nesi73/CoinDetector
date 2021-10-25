import os
import shutil
from pandas import DataFrame
import Augmentor

class generateDataset():
    def __init__(self):
        self.augmentor()

        folder = os.listdir("detected_coins/output/")
        
        #Division train val test
        train, val, test = self.split_data(folder)
        
        df_list = map(self.write_csv, train)
        df = DataFrame(df_list, columns=['Path', 'Class'])
        df.to_csv('database_coins_train.csv')

        df_list = map(self.write_csv, val)
        df = DataFrame(df_list, columns=['Path', 'Class'])
        df.to_csv('database_coins_val.csv')

        df_list = map(self.write_csv, test)
        df = DataFrame(df_list, columns=['Path', 'Class'])
        df.to_csv('database_coins_test.csv')

    def split_data(self, folder):
        train_len = int(len(folder)*0.7)
        val_len = int(len(folder)*0.2) + train_len
        train = folder[:train_len]
        val = folder[train_len:val_len]
        test = folder[val_len:]
        
        os.mkdir('detected_coins/train')
        os.mkdir('detected_coins/val')
        os.mkdir('detected_coins/test')
        os.mkdir('detected_coins/train/images')
        os.mkdir('detected_coins/val/images')
        os.mkdir('detected_coins/test/images')
        self.move_files(train, 'train/images/')
        self.move_files(val, 'val/images/')
        self.move_files(test, 'test/images/')
        return train, val, test

    def move_files(self, folder, name_folder): 
        for i in range((len(folder))):
            shutil.move('detected_coins/output/' + folder[i], 'detected_coins/'+ name_folder + '0000' + str(i + 1) + '.jpg')

    def write_csv(self, file):
        classes = 0
        return ['detected_coins/output/' + file, classes]

    def augmentor(self):
        p = Augmentor.Pipeline("detected_coins/")
        p.flip_left_right(probability=0.7)
        p.flip_top_bottom(probability=0.7)
        p.rotate90(probability=0.7)
        p.rotate270(probability=0.7)
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
        p.sample(1000)    

generate_dataset = generateDataset()