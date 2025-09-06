import pandas as pd

class DataLoader:
    @staticmethod
    def load_comments(files):
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def load_videos(file):
        return pd.read_csv(file)

    @staticmethod
    def standardize_comments(df):
        return df.rename(columns={"likeCount": "comment_likeCount"})

    @staticmethod
    def standardize_videos(df):
        return df.rename(columns={
            "likeCount": "video_likeCount",
            "viewCount": "video_viewCount",
            "commentCount": "video_commentCount",
            "favouriteCount": "video_favCount"
        })

    @staticmethod
    def merge(comments_df, videos_df):
        return comments_df.merge(videos_df, on="videoId", how="left")
