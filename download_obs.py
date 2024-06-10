from obs import ObsClient
import os
from io import BytesIO

access_key = 'AEFBJRJ1MOY1SWVLOULZ'
secret_key = 'BwRN5g6fSkUusFCWwDS7WRASBOdZy5XYYoldChMt'
endpoint = 'https://obs.ap-southeast-2.myhuaweicloud.com'
bucket_name = 'ai-model-bucket'

obs_client = ObsClient(
    access_key_id=access_key,
    secret_access_key=secret_key,
    server=endpoint
)


def download_image_to_bytes(download_path, obs_client, bucket_name):
    try:
        resp = obs_client.getObject(bucket_name, download_path, loadStreamInMemory=True)
        if resp.status < 300 and resp.body:
            return BytesIO(resp.body.buffer)
        else:
            print('Error: Unable to download image from OBS')
            return None
    except Exception as e:
        print(f'Error downloading image from OBS: {e}')
        return None
    
def list_all_objects(bucket_name, prefix):
    objects = []
    marker = None

    while True:
        response = obs_client.listObjects(bucket_name, prefix=prefix, marker=marker)

        if response.status < 300:
            objects.extend(response.body.contents)

            if response.body.is_truncated:
                marker = response.body.next_marker
            else:
                break
        else:
            print('Failed to list objects with status code:', response.status)
            break

    return objects

def download_object(get_objects):
    for content in get_objects:
        object_key = content.key
        # print(object_key)
        
        local_dir = os.path.dirname(object_key)
        
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            
        print('Local directory created:', local_dir)
        
        # Skip the directories
        if object_key.endswith('/'):
            print(f'Skipping directory {object_key}')
            continue
        # Download the image as bytes
        image_bytes = download_image_to_bytes(object_key, obs_client, bucket_name)
        if image_bytes:
            # Save the image locally
            
            with open(object_key, 'wb') as f:
                f.write(image_bytes.getbuffer())
            print(f'Download succeeded for {object_key}')
        else:
            print(f'Failed to download {object_key}')
            

if __name__ == "__main__":
    print(f"[INFO] Start download image from obs")
    current_dir = 'download_file/'
    prefix = 'segment_folder/segment_flavor/'
    response = obs_client.listObjects(bucket_name, prefix=prefix)
    get_objects = list_all_objects(bucket_name, prefix)
    download_object(get_objects)
    print(f"[INFO] Finished download image from obs")
    