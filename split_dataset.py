import pandas as pd
from download import split_dataframe
import os

df = pd.read_parquet("/home/user/main_storage/dataset/e6_dump.parquet")
# shuffle
df = df.sample(frac=1, random_state=42)
# remove unnecessary columns
df = df[["image_height", "image_width", "new_tag_string", "md5"]]
# create file name column
df["file"] = df["md5"] + ".webp"
# get available image file name
image_list = os.listdir("/home/user/main_storage/dataset/almost_all_e6_dataset")

df = df.loc[df["file"].isin(image_list)]
# convert tag list to text
df["new_tag_string"] = df["new_tag_string"].str.join(", ")

df_list = split_dataframe(df, 50)

for count, dataframe in enumerate(df_list):
    dataframe.to_csv(f"/home/user/main_storage/dataset/e6_dump_{count}.csv")

print()
