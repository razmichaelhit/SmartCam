from mega import Mega
import os
import shutil

def connectToMega():
    email = 'razmichaelhit@gmail.com'
    password = 'Hit123456!'
    mega = Mega()
    m = mega.login(email,password)
    return m
    
def createFolderInMega(m , name):
    m.create_folder(name)

def uploadFolderToMega(m, src, dst):
    for filename in os.listdir(src):
        filePath = src+'/'+filename
        folder_destination = m.find(dst)
        m.upload(filePath, folder_destination[0])
    
def CreateUploadDeleteOld(sourceFolder, megaFolder, folderToDelete):
    connection = connectToMega()
    createFolderInMega(connection, megaFolder)
    uploadFolderToMega(connection, sourceFolder, megaFolder)
    shutil.rmtree(folderToDelete)