from icrawler.builtin import BingImageCrawler
import os

NUM_IMAGES = 120

DATASET = {
    "brake_pad": "car brake pad",
    "brake_disc": "car brake disc rotor",
    "battery": "car battery",
    "spark_plug": "car spark plug",
    "air_filter": "car air filter",
    "shock_absorber": "car shock absorber",
    "ball_joint": "car ball joint suspension",
    "radiator": "car radiator",
    "starter_motor": "car starter motor",
    "ac_compressor": "car AC compressor"
}

BASE_FOLDER = "dataset"

for folder, keyword in DATASET.items():

    path = os.path.join(BASE_FOLDER, folder)

    os.makedirs(path, exist_ok=True)

    print("Downloading:", keyword)

    crawler = BingImageCrawler(
        downloader_threads=4,
        storage={"root_dir": path}
    )

    crawler.crawl(
        keyword=keyword,
        max_num=NUM_IMAGES
    )

print("Dataset download complete")