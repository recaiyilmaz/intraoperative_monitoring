# THIS SCRIPT SYNCHS A LOCAL FOLDER AND A CLOUD FOLDER,
# MAKING SURE ALL VIDEO AND PERFORMANCE DATA IS UPLOADED TO THE CLOUD

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

LOCAL_FOLDER = 'Video_Output'
CLOUD_FOLDER_ID = '1Nq7IY1eNP3ppuxQ7OT7hcjllAiwZz1G7'   # example cloud ID
SERVICE_ACCOUNT_FILE = '.JSON'  # link to the service account file
SCOPES = ['https://www.googleapis.com/auth/drive']

# Authenticate and create the service
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

def list_files_in_drive_folder(service, folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, pageSize=1000, fields="files(id, name)").execute()
    items = results.get('files', [])
    return {item['name']: item['id'] for item in items}

def upload_file_to_drive(service, file_path, folder_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'File {file["id"]} uploaded as {file_path}')

def main():
    # List files in the cloud folder
    cloud_files = list_files_in_drive_folder(service, CLOUD_FOLDER_ID)

    # Scan the local folder and upload files not present in the cloud folder
    for root, _, files in os.walk(LOCAL_FOLDER):
        for file_name in files:
            if file_name not in cloud_files:
                file_path = os.path.join(root, file_name)
                upload_file_to_drive(service, file_path, CLOUD_FOLDER_ID)
            else:
                print(f'File {file_name} already exists in the cloud.')

if __name__ == '__main__':
    main()
