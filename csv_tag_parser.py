import pandas as pd
import os


# global var

# debug mode
debug = True
tag_index_test = True
debug_random_seed = 42
enable_tag_threshold = False
# set this to 0 if you want to keep everything
low_tag_count_threshold = 10
low_artist_post_count_threshold = 10
stripped_meta_tags = [
    "description",
    "text",
    "audio",
    "version",
    "\d+:\d+",
    "^\d{4}$",
    r"\bsource\b",
]


def create_tag_index(df: pd.DataFrame, col_id: str, tag_col: str) -> pd.DataFrame:
    """
    Create a tag index DataFrame based on the provided DataFrame.
    can also be used to revert it back to tag list

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        col_id (str): The name of the column in df that represents the unique identifier for each row.
        tag_col (str): The name of the column in df that contains the tags.

    Returns:
        pd.DataFrame: A DataFrame representing the tag index, with tags as the index and col_id as the column.

    Example:
        >>> df = pd.DataFrame({'id': [1, 2, 3], 'tags': [['tag1', 'tag2'], ['tag2', 'tag3'], None]})
        >>> df_tags = create_tag_index(df, 'id', 'tags')
        >>> df_tags
        tag_col  col_id
        tag1     [1]
        tag2     [1, 2]
        tag3     [2]
        >>> create_tag_index(df_tags, 'id', 'tags')
        id  tags
        1  [tag1, tag2]
        2  [tag2, tag3]
    """
    tag_index = {}

    # Iterate through each row of the dataset using itertuples()
    for index, row in df.iterrows():
        id = row[col_id]
        tags = row[tag_col]

        if tags != None:
            # Add id to the list of each tag in the dictionary
            for tag in tags:
                if tag in tag_index:
                    tag_index[tag].append(id)
                else:
                    tag_index[tag] = [id]

    # Convert the tag_index dictionary to a DataFrame
    tag_index_df = pd.DataFrame(list(tag_index.items()), columns=[tag_col, col_id])
    return tag_index_df


def separate_tag_string_as_comma_separated(
    df: pd.DataFrame, tag_col: str
) -> pd.DataFrame:
    """
    Separates a tag string in a specified column of a DataFrame into multiple tags,
    treating commas as separators. Replaces spaces with commas and underscores with spaces
    before splitting the string.

    Args:
        df (pd.DataFrame): The input DataFrame.
        tag_col (str): The name of the column containing the tag string.

    Returns:
        pd.DataFrame: The DataFrame with the tag string separated into multiple tags.

    Example:
        >>> df = pd.DataFrame({'Tags': ['apple orange banana', 'grape kiwi']})
        >>> separated_df = separate_tag_string_as_comma_separated(df, 'Tags')
        >>> print(separated_df)
                    Tags
        0  [apple, orange, banana]
        1          [grape, kiwi]
    """
    df[tag_col] = df[tag_col].str.replace(" ", ",")
    df[tag_col] = df[tag_col].str.replace("_", " ")
    df[tag_col] = df[tag_col].str.split(",")

    return df


def main():
    # NOTE: this entire function here is to "strips" unnecessary tags NOT
    # to delete, filter, or drop post

    # relative current dir of this script
    current_dir = os.getcwd()

    # tags
    tag_lut_dir = os.path.join(current_dir, "tags-2023-05-10.csv")
    tag_lut = pd.read_csv(tag_lut_dir)

    # e6 db dump
    e6_dump = os.path.join(current_dir, "posts-2023-05-10.parquet")
    e6_dump = pd.read_parquet(e6_dump)

    if debug:
        e6_dump = e6_dump.sample(n=10**4, random_state=debug_random_seed)

    # strip underscores and convert tag string into a list of tags for each post
    e6_dump = separate_tag_string_as_comma_separated(df=e6_dump, tag_col="tag_string")
    # strips underscores from tags
    tag_lut["name"] = tag_lut["name"].str.replace("_", " ")
    print()

    # this indexing is slow, you might want to run this on multiple cores
    # or not? this just took like 10 minutes to run through all 4m rows
    tag_to_index = create_tag_index(df=e6_dump, col_id="id", tag_col="tag_string")

    # count index occurrence in the e6 database dump
    tag_to_index["tag_count"] = tag_to_index["id"].apply(len)

    # strip tags that below certain threshold
    # disable this during debugging because artist tags
    # will have low tag count
    if enable_tag_threshold:
        # filter rare tags
        tag_to_index = tag_to_index[tag_to_index["tag_count"] > low_tag_count_threshold]
        tag_lut = tag_lut[tag_lut["post_count"] > low_tag_count_threshold]

    # ===[get artist name and prepend "by"]=== #

    # store list of artist name
    list_of_artist = tag_lut.loc[tag_lut["category"] == 1, "name"].to_list()

    # boolean mask to select all artist tags
    select_artist = tag_to_index["tag_string"].isin(list_of_artist)

    # {artist_name} >> "by {artist_name}"
    tag_to_index.loc[select_artist, "tag_string"] = (
        "by " + tag_to_index.loc[select_artist, "tag_string"]
    )

    # ===[strip lore tag]=== #
    # get lore tag from tag_lut
    lore_tag = tag_lut[tag_lut.category == 8]["name"].to_list()
    #  exclude lore tag
    tag_to_index = tag_to_index[~tag_to_index["tag_string"].isin(lore_tag)]

    # ===[bulk strip meta tags]=== #
    meta_tags = tag_lut[tag_lut["category"] == 7]

    for stripped_meta_tag in stripped_meta_tags:
        meta_tags = meta_tags[~meta_tags["name"].str.contains(stripped_meta_tag)]

    # select inverse of the meta tags
    dropped_meta_tags = tag_lut[tag_lut["category"] == 7]
    dropped_meta_tags = dropped_meta_tags[
        ~dropped_meta_tags["name"].isin(meta_tags["name"].to_list())
    ]
    dropped_meta_tags = dropped_meta_tags["name"].to_list()

    #  exclude some meta tags
    tag_to_index = tag_to_index[~tag_to_index["tag_string"].isin(dropped_meta_tags)]

    # run test to compare if original data is equal after index transformation
    if tag_index_test:
        # create sets of tags for transformed data
        index_to_tag = create_tag_index(
            df=tag_to_index, col_id="tag_string", tag_col="id"
        )
        index_to_tag = index_to_tag.set_index("id").sort_index()
        index_to_tag = index_to_tag["tag_string"].apply(set)

        # create sets of tags for untransformed data
        temporary_e6_dump = e6_dump[["id", "tag_string"]].set_index("id").sort_index()
        # BUG! there's None or numpy NA in here
        # TODO! find numpy na check and replace it with empty array instead
        temporary_e6_dump = temporary_e6_dump["tag_string"].apply(set)

        # compare
        comparison = (temporary_e6_dump == index_to_tag).value_counts()
        print(comparison)

    # revert back indexing
    index_to_tag = create_tag_index(df=tag_to_index, col_id="tag_string", tag_col="id")
    index_to_tag = index_to_tag.rename(columns={"tag_string": "new_tag_string"})
    index_to_tag = index_to_tag.set_index("id")

    e6_dump = e6_dump.set_index("id")

    # concat as new column so the original column stays and can be used for cross checking
    e6_dump = pd.concat([e6_dump, index_to_tag], axis=1)

    print()


main()
