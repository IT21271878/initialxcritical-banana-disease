import os
import shutil
import subprocess

def prepare_dataset():
    # 1. Clean up dataset directory
    base_dir = r"c:\Code\banana-disease-app\backend\dataset"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)
    
    # 2. Extract Dataset.rar using tar (some errors might happen but most files succeed)
    print("Extracting Dataset.rar...")
    rar_path = r"c:\Code\banana-disease-app\backend\Dataset.rar"
    subprocess.run(["tar", "-xf", rar_path, "-C", base_dir], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 3. Restructure
    # The expected structures from tar extract:
    # dataset/Panama Disease/Panama infected/   => Panama Disease
    # dataset/Panama Disease/Healthy/           => Healthy
    # dataset/Sigatoka Disease/sigatoka data/   => Sigatoka Disease
    # dataset/Sigatoka Disease/healthy leaf/    => Healthy
    # dataset/Sigatoka Disease/not detected folder/ => not detected
    
    clean_dir = r"c:\Code\banana-disease-app\backend\clean_dataset"
    if os.path.exists(clean_dir):
        shutil.rmtree(clean_dir, ignore_errors=True)
    
    os.makedirs(os.path.join(clean_dir, "Panama Disease"), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, "Sigatoka Disease"), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, "Healthy"), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, "not detected"), exist_ok=True)
    
    def copy_files(src, dst):
        if not os.path.exists(src):
            print(f"Warning: Source not found {src}")
            return
        for file in os.listdir(src):
            src_file = os.path.join(src, file)
            # handle images
            if os.path.isfile(src_file) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # to avoid naming collisions
                dst_file = os.path.join(dst, f"{os.path.basename(os.path.dirname(src))}_{file}")
                shutil.copy2(src_file, dst_file)

    copy_files(os.path.join(base_dir, "Panama Disease", "Panama infected"), os.path.join(clean_dir, "Panama Disease"))
    copy_files(os.path.join(base_dir, "Panama Disease", "Healthy"), os.path.join(clean_dir, "Healthy"))
    copy_files(os.path.join(base_dir, "Sigatoka Disease", "sigatoka data"), os.path.join(clean_dir, "Sigatoka Disease"))
    copy_files(os.path.join(base_dir, "Sigatoka Disease", "healthy leaf"), os.path.join(clean_dir, "Healthy"))
    copy_files(os.path.join(base_dir, "Sigatoka Disease", "not detected folder"), os.path.join(clean_dir, "not detected"))
    
    # Replace the old dataset with clean dataset
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.move(clean_dir, base_dir)
    print("Dataset restructured successfully!")

if __name__ == "__main__":
    prepare_dataset()
