from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()           
drive = GoogleDrive(gauth)  

def upload_file(filename, parent_id):
    gfile = drive.CreateFile({'parents': [{'id': parent_id}]})
    gfile.SetContentFile(filename)
    gfile.Upload()
    
