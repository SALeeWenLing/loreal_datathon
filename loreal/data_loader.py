import pandas as pd

class DataLoader:
    @staticmethod
    def load_comments(files):
        '''
        - Loads comments data from multiple CSV files.
        - Returns a concatenated DataFrame of all comments.
        '''
        dfs = [pd.read_csv(f) for f in files]
        print(f"Loaded {len(dfs)} comment files with total {sum(len(df) for df in dfs)} rows.")
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def load_videos(file):
        '''
        - Loads video data from a CSV file.
        - Returns a DataFrame of video metadata.
        '''
        print(f"Loading video data from {file}")
        return pd.read_csv(file)

    @staticmethod
    def standardize_comments(df):
        '''
        - Standardizes comment DataFrame column names so that there is no overlap with video columns.
        - Renames "likeCount" to "comment_likeCount".
        '''
        print(f"Standardizing comment DataFrame columns")
        return df.rename(columns={"likeCount": "comment_likeCount"})

    @staticmethod
    def standardize_videos(df):
        '''
        - Standardizes video DataFrame column names.
        '''
        print(f"Standardizing video DataFrame columns")
        return df.rename(columns={
            "likeCount": "video_likeCount",
            "viewCount": "video_viewCount",
            "commentCount": "video_commentCount",
            "favouriteCount": "video_favCount"
        })

    @staticmethod
    def merge(comments_df, videos_df):
        '''
        - Merges comments DataFrame with videos DataFrame on "videoId".
        '''
        print(f"Merging comments and videos DataFrames")
        return comments_df.merge(videos_df, on="videoId", how="left")
