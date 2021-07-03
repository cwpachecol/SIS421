import os
from os.path import splitext

path_images = "D:\\Repositorios\\SIS421\\ML\\Projects\\Utensilios\\Data\\JPEGImages"
included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
file_names = [splitext(fn)[0] for fn in os.listdir(path_images) if any(fn.endswith(ext) for ext in included_extensions)]

print(len(file_names))

f = open("D:\\Repositorios\\SIS421\ML\\Projects\\Utensilios\\Data\\train.txt",'w')
for ele in file_names[:160]:
    f.write(ele + '\n')
f.close()

f = open("D:\\Repositorios\\SIS421\ML\\Projects\\Utensilios\\Data\\test.txt",'w')
for ele in file_names[160:]:
    f.write(ele + '\n')
f.close()