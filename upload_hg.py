import pandas as pd
import os
import math
import subprocess
from tqdm import tqdm
from huggingface_hub import HfApi, login
from multiprocessing import Pool
from data_reader import read_general
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Configuration
os.environ['HF_TOKEN'] = ''
PREFIX = '/mnt/petrelfs/zhaoshitian/data/Synthesized_data/synthetic_annotations/tmp'
CSV_FILE = '/home/ubuntu/goosedata/captioned_data/new_captioned_data_9_20_24.csv'  # Replace with your CSV file path
BATCH_SIZE = 1000
REPO_ID = 'stzhao/structure_anno'  # Replace with your username and dataset nametqdm()
start_idx = 0

def process_data(data, save_image_folder, new_data_list, lock):
    try:
        image_path = data['meta_data']['image_url']
        img_name = "_".join(image_path.split("//", -1)[-1].split("/", -1))
        if not os.path.exists(os.path.join(save_image_folder, f"{img_name}")):
            image_bytes_io = read_general(image_path)
            image = Image.open(image_bytes_io)
            image.save(os.path.join(save_image_folder, f"{img_name}"))
        # 使用锁来保证线程安全地修改 new_data_list
        with lock:
            data['meta_data']['image_url'] = os.path.join(save_image_folder, f"{img_name}")
            new_data_list.append(data)
        print(img_name)
    except:
        print("something wrong")

def move_data_to_local(all_data_path, save_image_folder="/mnt/petrelfs/zhaoshitian/data/Synthesized_data/synthetic_annotations/processed_images_from_laion"):
    data_list = json.load(open(all_data_path, "r"))
    new_data_list = []
    lock = Lock()
    
    # 使用线程池来并发处理数据
    with ThreadPoolExecutor() as executor:
        # 提交所有任务并跟踪进度
        list(tqdm(executor.map(lambda data: process_data(data, save_image_folder, new_data_list, lock), data_list), total=len(data_list)))

    # 保存新的数据列表到文件
    save_path = "/mnt/petrelfs/zhaoshitian/data/Synthesized_data/synthetic_annotations/new_all_annotation_laion.json"
    with open(save_path, "w") as f:
        json.dump(new_data_list, f, indent=4)

def curate_image_list(json_path):
    data_list = json.load(open(json_path, "r"))
    image_url_list = []
    for data in data_list:
        image_url = data['meta_data']['image_url']
        image_url_list.append(image_url)
    return image_url_list

# Function to check if a file exists (used for parallel processing)
def check_file(img_path):
    # img_path = os.path.join(PREFIX, img_name)
    # exists = os.path.exists(img_path)
    save_image_folder = PREFIX
    try:
        # img_name = "_".join(img_path.split("//", -1)[-1].split("/", -1))
        img_name = img_path.split("/", -1)[-1]
        image = Image.open(img_path)
        image.save(os.path.join(save_image_folder, f"{img_name}"))
        exists = True
        print(img_name)
    except:
        exists = False
    print(exists)
    return img_name if exists else None

# # Step 1: Read image names from the CSV file
# df = pd.read_csv(CSV_FILE)
# # Assuming the CSV has a column named 'image_name' with the image names
# image_names = df['image_name'].tolist()

def main():
    # Configuration
    os.environ['HF_TOKEN'] = ''
    PREFIX = '/mnt/petrelfs/zhaoshitian/data/Synthesized_data/synthetic_annotations/tmp'
    CSV_FILE = '/home/ubuntu/goosedata/captioned_data/new_captioned_data_9_20_24.csv'  # Replace with your CSV file path
    BATCH_SIZE = 50000
    REPO_ID = 'stzhao/structure_anno'  # Replace with your username and dataset nametqdm()
    start_idx = 0

    json_path = "/mnt/petrelfs/zhaoshitian/data/Synthesized_data/synthetic_annotations/new_all_annotation_laion.json"
    image_names = curate_image_list(json_path)

    total_images = len(image_names)
    num_batches = math.ceil(total_images / BATCH_SIZE)

    # Step 2: Set up Hugging Face API
    token = os.getenv('HF_TOKEN')

    if not token:
        token = input('Please enter your Hugging Face token: ')

    # Log in to Hugging Face Hub
    login(token=token)
    api = HfApi()

    # Step 3: Process images in batches
    for batch_num in range(start_idx, num_batches):
        if batch_num <= 2 or batch_num == 4:
            continue
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, total_images)
        batch_image_names = image_names[start_idx:end_idx]

        archive_name = f'images_batch_{batch_num + 1}.tar'
        file_list_name = f'file_list_{batch_num + 1}.txt'

        print(f'\nPreparing file list for batch {batch_num + 1}...')

        # Check file existence in parallel
        with Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap(check_file, batch_image_names), total=len(batch_image_names), desc=f'Checking files in Batch {batch_num + 1}'))

        # Filter out None results (non-existent files) and write to file list
        existing_files = [img for img in results if img is not None]

        if not existing_files:
            print(f'No existing files found in batch {batch_num + 1}. Skipping...')
            continue
        
        # Write the list of existing files to a text file
        with open(file_list_name, 'w') as f_list:
            for img_name in existing_files:
                img_path = os.path.join(PREFIX, img_name)
                # img_path = img_name
                f_list.write(f'{img_path}\n')

        # Create the tar archive using the system's tar command without compression
        print(f'Creating archive {archive_name} using system tar command...')
        subprocess.run(['tar', '-cf', archive_name, '-T', file_list_name], check=True)

        # Remove the file list
        os.remove(file_list_name)

        # Step 4: Upload the archive to Hugging Face
        print(f'Uploading {archive_name} to Hugging Face...')
        try:
            api.upload_file(
                path_or_fileobj=archive_name,
                path_in_repo=f'images_laion/{archive_name}',
                repo_id=REPO_ID,
                repo_type='dataset',
            )
            print(f'Successfully uploaded {archive_name}.')
        except Exception as e:
            print(f'Failed to upload {archive_name}: {e}')

        # Optional: Remove the local archive to save disk space
        os.remove(archive_name)

if __name__ == "__main__":
    # all_data_path = "/mnt/petrelfs/zhaoshitian/data/Synthesized_data/synthetic_annotations/all_annotation_laion.json"
    # move_data_to_local(all_data_path)
    main()
