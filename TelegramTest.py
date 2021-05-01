from PIL import Image
import glob
import os, shutil


def ResizeAllPicturesInFOlder(path):
    lst_imgs = [i for i in glob.glob(path+"/*.png")]
    
    # It creates a folder called ltl if does't exist
    if not "Resized" in os.listdir():
        os.mkdir("Resized")  
    print(lst_imgs)
    for i in lst_imgs:
        img = Image.open(i)
        img = img.resize((500, 500), Image.ANTIALIAS)
        img.save(i[:-4] +".png")
    print("Done")

def deleteAllFolder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# ResizeAllPicturesInFOlder('/home/rami/Desktop/Project/Pictures/backup/2021-05-01 19:58:16.873240')

deleteAllFolder('/home/rami/Desktop/Project/Pictures/backup/2021-05-01 19:58:16.873240')