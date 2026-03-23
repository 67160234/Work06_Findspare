from icrawler.builtin import BingImageCrawler
from PIL import Image
import os
import shutil
import time

shop_prefix = "suk"

parts = [
    "ac_compressor",
    "air_filter",
    "ball_joint",
    "battery",
    "brake_disc",
    "brake_pad",
    "radiator",
    "shock_absorber",
    "spark_plug",
    "starter_motor"
]

save_folder = "shop_parts"
os.makedirs(save_folder, exist_ok=True)

for part in parts:

    print("Downloading:", part)

    temp = "temp"

    if os.path.exists(temp):
        shutil.rmtree(temp, ignore_errors=True)

    os.makedirs(temp)

    keyword = part.replace("_"," ") + " car part"

    crawler = BingImageCrawler(storage={'root_dir': temp})
    crawler.crawl(keyword=keyword, max_num=10)

    best_image = None
    best_resolution = 0

    for file in os.listdir(temp):

        path = os.path.join(temp,file)

        try:
            # ปิดไฟล์อัตโนมัติ
            with Image.open(path) as img:
                width, height = img.size
                resolution = width * height

            if resolution > best_resolution:
                best_resolution = resolution
                best_image = path

        except:
            continue

    if best_image:

        new_name = f"{shop_prefix}_{part}.jpg"
        new_path = os.path.join(save_folder,new_name)

        shutil.move(best_image,new_path)

        print("Saved:", new_name)

    # รอให้ crawler thread ปิดไฟล์
    time.sleep(1)

    shutil.rmtree(temp, ignore_errors=True)

print("Done")