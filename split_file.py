import os
import re
import shutil

PATH = os.path.abspath("D:\project_biometrika\dataset")


value_dir = os.listdir(PATH)
class_dir = os.listdir(PATH+"\\train")
def main():

    if('train' not in value_dir):
        os.mkdir(PATH+"\\train")
        print("Train folder Created")

    if('test' not in value_dir):
        os.mkdir(PATH+"\\test")
        print("Test folder Created")

    print("looking for data...")


    for folders in class_dir:

        try:
            dirs = os.listdir(PATH+"\\{0}\\{1}".format('train',folders))

            train_data, test_data =  split_list(dirs,0.2)   
            x = folders.split("_")

          
            if(os.path.exists(PATH+"\\{0}\\{1}".format("train",x[0])) is False):
                os.mkdir(PATH+"\\{0}\\{1}".format("train",x[0]))
                print("{} folder created".format(x[0]))

            if(os.path.exists(PATH+"\\{0}\\{1}".format("test",x[0])) is False):
                os.mkdir(PATH+"\\{0}\\{1}".format("test",x[0]))
                print("{} folder created".format(x[0]))


            for a in test_data:
               # shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
                shutil.move(PATH+"\\{0}\\{1}\\{2}".format('train',folders,a),PATH+"\\{0}\\{1}\\{2}".format('test',folders,a))
            # shutil.copy2(PATH+"\\{0}\\{1}".format("train",file_train),loc+"\\Train\\{}".format(file_train))

            pass
        except Exception as e:
            print(e)
          
    pass

def split_list(list_val,split_val = 0.2):
    print("split val to test and train")
    length_train = round(len(list_val)*split_val)

    return list_val[length_train:],list_val[:length_train]


if __name__ == "__main__":
    main()
    pass