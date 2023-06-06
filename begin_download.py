from download import *
import pandas as pd
import os
from threading import Thread
import multiprocessing
from PIL import Image
import PIL
import gc


# constant variable, extract this out please
current_dir = os.getcwd()
filename = "posts-2023-05-10.csv"
save_path = "/home/user/project-fur/e6_dump/almost_all_e6_dataset"
# save_path = os.path.join(current_dir, save_path)
number_of_workers = 20
numb_of_threads_each_worker = 10
excluded_tags = ["cub", "gore", "animated"]
do_download = True
check_integrity = True


# wrap in function so the gc can collect stuff


def filter():
    # only grab png and jpg file from e6
    e6_dump_data = pd.read_csv(os.path.join(current_dir, filename))
    data_currated_jpg = e6_dump_data[e6_dump_data["file_ext"] == ("jpg")]
    data_currated_png = e6_dump_data[e6_dump_data["file_ext"] == ("png")]
    data_currated = pd.concat([data_currated_jpg, data_currated_png])
    del data_currated_jpg
    del data_currated_png

    # spliting tag_string into list of tags
    data_currated["tag_lists"] = data_currated["tag_string"].str.split(" ")
    data_currated = data_currated.dropna(subset=["tag_lists"])

    # a fucntion to be applied (should be lambda function but oh well)
    def is_contain(list_str: list, text: str) -> bool:
        """check whether the text is exist in a list"""
        return text in list_str

    # remove legally / politically concerning data
    for excluded_tag in excluded_tags:
        data_currated = data_currated[
            data_currated["tag_lists"].apply(
                is_contain, text=excluded_tag) == False
        ]

    data_currated = data_currated[["md5", "file_ext"]]
    return data_currated


def download(dataset: pd.DataFrame, save_path: str):
    for index, sample in dataset.iterrows():

        attempt = 0
        while attempt < 5:
            try:
                # generate url
                file_ext = sample["file_ext"]
                md5 = sample["md5"]
                url = f"https://static1.e621.net/data/{md5[0:2]}/{md5[2:4]}/{md5}.{file_ext}"

                user_agent_message = (
                    f"heya! this is lodestone from furry diffusion server, "
                    f"i need to rebuild my dataset for training, "
                    f"please let me know if this bot is pulling data to fast "
                    f"im pulling with 180 concurent thread"
                )

                # download image as PIL object
                image = stream_image(
                    url, user_agent=user_agent_message, threshold_size=0
                )

                image = rescale_image(image, 1024)

                save_webp_without_alpha(
                    image, os.path.join(save_path, f"{md5}.webp"), quality=70
                )

                break
            except Exception as e:
                print(
                    f"failed downloading {index} reason:{e} ... retrying {attempt}/10"
                )
                attempt += 1
        else:
            print(f"download attempt exceeded skipping {index}")
            continue


def multithread_download(
    df: pd.DataFrame, save_path: str, number_of_workers: int = 10
) -> None:
    # function tp be executed as threads
    split_df = split_dataframe(df, number_of_workers)

    threads = []
    for df in split_df:
        thread_instance = Thread(target=download, args=[df, save_path])
        threads.append(thread_instance)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def check_error(filename: str) -> list:
    list_broken_image = []
    try:
        im = Image.open(filename)
        im.verify()
        im.close()
        im = Image.open(filename)
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        im.close()
    except Exception as e:
        print(f"image error {filename}: {e}")
        list_broken_image.append(filename)
    return list_broken_image


def main():
    data_currated = filter()
    if do_download:
        gc.collect()
        split_df = split_dataframe(data_currated, number_of_workers)
        del data_currated

        args = [(df,) + (save_path,) + (numb_of_threads_each_worker,)
                for df in split_df]

        with multiprocessing.Pool(processes=number_of_workers) as pool:
            results = pool.starmap(multithread_download, args)

    # check integrity
    if check_integrity:
        list_image = os.listdir(save_path)
        list_image = [os.path.join(save_path, image) for image in list_image]

        with multiprocessing.Pool(processes=number_of_workers) as pool:
            results = pool.map(check_error, list_image)
            print(results)

        flat_list = []
        for sublist in results:
            for element in sublist:
                flat_list.append(element)

        broken_image = [text.split("/")[-1] for text in flat_list]
        broken_image = [md5.split(".")[0] for md5 in broken_image]
        print(broken_image)


main()
