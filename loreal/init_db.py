import duckdb
from pathlib import Path
import sys

DATASET_DIR = Path("dataset_cleaned")
DB_FILE = "loreal_datathon.duckdb"


def setup_database(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        kind VARCHAR,
        channelId VARCHAR,
        videoId BIGINT PRIMARY KEY,
        title VARCHAR,
        description VARCHAR,
        defaultLanguage VARCHAR,
        defaultAudioLanguage VARCHAR,
        contentDuration VARCHAR,
        viewCount BIGINT,
        likeCount BIGINT,
        favouriteCount BIGINT,
        commentCount BIGINT,
        publishedAt TIMESTAMP
    );

    CREATE SEQUENCE IF NOT EXISTS tag_seq START 1;
    CREATE TABLE IF NOT EXISTS tags (
        tag_id INTEGER PRIMARY KEY DEFAULT nextval('tag_seq'),
        tag_name VARCHAR UNIQUE
    );

    CREATE TABLE IF NOT EXISTS video_tags (
        videoId BIGINT,
        tag_id INTEGER,
        PRIMARY KEY (videoId, tag_id),
        FOREIGN KEY (videoId) REFERENCES videos(videoId),
        FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
    );

    CREATE SEQUENCE IF NOT EXISTS category_seq START 1;
    CREATE TABLE IF NOT EXISTS topic_categories (
        category_id INTEGER PRIMARY KEY DEFAULT nextval('category_seq'),
        category_url VARCHAR UNIQUE
    );

    CREATE TABLE IF NOT EXISTS video_categories (
        videoId BIGINT,
        category_id INTEGER,
        PRIMARY KEY (videoId, category_id),
        FOREIGN KEY (videoId) REFERENCES videos(videoId),
        FOREIGN KEY (category_id) REFERENCES topic_categories(category_id)
    );

    CREATE TABLE IF NOT EXISTS comments (
        kind VARCHAR,
        commentId BIGINT PRIMARY KEY,
        parentCommentId BIGINT,
        channelId VARCHAR,
        videoId BIGINT,
        authorId VARCHAR,
        textOriginal VARCHAR,
        likeCount INTEGER,
        publishedAt TIMESTAMP,
        updatedAt TIMESTAMP,
        FOREIGN KEY (videoId) REFERENCES videos(videoId)
    );

    -- store orphan/skipped comments for inspection
    CREATE TABLE IF NOT EXISTS orphan_comments AS SELECT * FROM comments WHERE FALSE;
    """)
    # indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_videos_channelId ON videos(channelId);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_videos_publishedAt ON videos(publishedAt);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_videoId ON comments(videoId);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_authorId ON comments(authorId);")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_publishedAt ON comments(publishedAt);")

    print("Database schema ready.")


def import_videos(conn: duckdb.DuckDBPyConnection, videos_parquet_path: str):
    print(f"Importing videos from {videos_parquet_path} ...")
    conn.execute(
        f"CREATE TEMPORARY TABLE temp_videos AS SELECT * FROM read_parquet('{videos_parquet_path}');")

    # Deduplicate by videoId (keep one row per videoId)
    conn.execute("""
    CREATE TEMPORARY TABLE temp_videos_deduped AS
    SELECT * EXCLUDE (row_number)
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY CAST(videoId AS BIGINT) ORDER BY publishedAt NULLS LAST) AS row_number
        FROM temp_videos
    )
    WHERE row_number = 1;
    """)
    conn.execute("DROP TABLE temp_videos;")
    conn.execute("ALTER TABLE temp_videos_deduped RENAME TO temp_videos;")

    # Insert videos (avoid duplicates already in videos)
    conn.execute("""
    INSERT INTO videos (kind, channelId, videoId, title, description, defaultLanguage,
                        defaultAudioLanguage, contentDuration, viewCount, likeCount,
                        favouriteCount, commentCount, publishedAt)
    SELECT kind, channelId, CAST(videoId AS BIGINT), title, description, defaultLanguage,
           defaultAudioLanguage, contentDuration,
           NULLIF(CAST(viewCount AS BIGINT), 0) as viewCount,
           NULLIF(CAST(likeCount AS BIGINT), 0) as likeCount,
           NULLIF(CAST(favouriteCount AS BIGINT), 0) as favouriteCount,
           NULLIF(CAST(commentCount AS BIGINT), 0) as commentCount,
           publishedAt
    FROM temp_videos
    WHERE CAST(videoId AS BIGINT) NOT IN (SELECT videoId FROM videos);
    """)

    # Insert distinct tags (only new tag names)
    conn.execute("""
    INSERT INTO tags (tag_name)
    SELECT tag_name FROM (
      SELECT DISTINCT NULLIF(TRIM(BOTH '"' FROM j.value::VARCHAR), '') AS tag_name
      FROM temp_videos
      CROSS JOIN json_each(temp_videos.tags) AS j(key, value)
      WHERE temp_videos.tags IS NOT NULL AND temp_videos.tags != '[]'
    ) t
    WHERE NOT EXISTS (
      SELECT 1 FROM tags WHERE tags.tag_name = t.tag_name
    );
    """)

    # Insert distinct (videoId, tag_id) pairs — skip existing ones
    conn.execute("""
    INSERT INTO video_tags (videoId, tag_id)
    SELECT vt.vid, t.tag_id
    FROM (
      SELECT DISTINCT CAST(v.videoId AS BIGINT) AS vid,
                      NULLIF(TRIM(BOTH '"' FROM j.value::VARCHAR), '') AS tag_name
      FROM temp_videos v
      CROSS JOIN json_each(v.tags) AS j(key, value)
      WHERE v.tags IS NOT NULL AND v.tags != '[]'
    ) vt
    JOIN tags t ON t.tag_name = vt.tag_name
    WHERE NOT EXISTS (
      SELECT 1 FROM video_tags vt2 WHERE vt2.videoId = vt.vid AND vt2.tag_id = t.tag_id
    );
    """)

    # topic categories: insert distinct categories
    conn.execute("""
    INSERT INTO topic_categories (category_url)
    SELECT category_url FROM (
      SELECT DISTINCT NULLIF(TRIM(BOTH '"' FROM j.value::VARCHAR), '') AS category_url
      FROM temp_videos
      CROSS JOIN json_each(temp_videos.topicCategories) AS j(key, value)
      WHERE temp_videos.topicCategories IS NOT NULL AND temp_videos.topicCategories != '[]'
    ) c
    WHERE NOT EXISTS (
      SELECT 1 FROM topic_categories WHERE topic_categories.category_url = c.category_url
    );
    """)

    # Insert distinct (videoId, category_id) pairs — skip existing ones
    conn.execute("""
    INSERT INTO video_categories (videoId, category_id)
    SELECT vc.vid, c.category_id
    FROM (
      SELECT DISTINCT CAST(v.videoId AS BIGINT) AS vid,
                      NULLIF(TRIM(BOTH '"' FROM j.value::VARCHAR), '') AS category_url
      FROM temp_videos v
      CROSS JOIN json_each(v.topicCategories) AS j(key, value)
      WHERE v.topicCategories IS NOT NULL AND v.topicCategories != '[]'
    ) vc
    JOIN topic_categories c ON c.category_url = vc.category_url
    WHERE NOT EXISTS (
      SELECT 1 FROM video_categories vc2 WHERE vc2.videoId = vc.vid AND vc2.category_id = c.category_id
    );
    """)

    conn.execute("DROP TABLE IF EXISTS temp_videos;")
    print("Videos import completed.")


def import_comments(conn: duckdb.DuckDBPyConnection, comments_parquet_path: str):
    print(f"Importing comments from {comments_parquet_path} ...")
    conn.execute(
        f"CREATE TEMPORARY TABLE temp_comments AS SELECT * FROM read_parquet('{comments_parquet_path}');")

    # Deduplicate comments by commentId using TRY_CAST (keep the latest by publishedAt if present)
    conn.execute("""
    CREATE TEMPORARY TABLE temp_comments_deduped AS
    SELECT * EXCLUDE (row_num)
    FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY TRY_CAST(commentId AS BIGINT)
            ORDER BY publishedAt NULLS LAST
        ) AS row_num
        FROM temp_comments
    )
    WHERE row_num = 1;
    """)
    conn.execute("DROP TABLE temp_comments;")
    conn.execute("ALTER TABLE temp_comments_deduped RENAME TO temp_comments;")

    # Count orphans (comments that reference missing or malformed videoId)
    orphan_count = conn.execute("""
    SELECT COUNT(*)::BIGINT AS orphan_count
    FROM temp_comments c
    LEFT JOIN videos v ON TRY_CAST(c.videoId AS BIGINT) = v.videoId
    WHERE TRY_CAST(c.videoId AS BIGINT) IS NULL OR v.videoId IS NULL;
    """).fetchone()

    if orphan_count and orphan_count[0] > 0:
        print(
            f"Found {orphan_count[0]} orphan comments in this batch (they will be moved to orphan_comments).")

    # Move orphan comments to orphan_comments for inspection
    conn.execute("""
    INSERT INTO orphan_comments
    SELECT c.kind,
           TRY_CAST(c.commentId AS BIGINT) AS commentId,
           TRY_CAST(c.parentCommentId AS BIGINT) AS parentCommentId,
           c.channelId,
           TRY_CAST(c.videoId AS BIGINT) AS videoId,
           c.authorId,
           c.textOriginal,
           TRY_CAST(c.likeCount AS INTEGER) AS likeCount,
           c.publishedAt,
           c.updatedAt
    FROM temp_comments c
    LEFT JOIN videos v ON TRY_CAST(c.videoId AS BIGINT) = v.videoId
    WHERE TRY_CAST(c.videoId AS BIGINT) IS NULL OR v.videoId IS NULL;
    """)

    # Insert comments that reference existing videos
    conn.execute("""
    INSERT INTO comments (kind, commentId, parentCommentId, channelId, videoId,
                          authorId, textOriginal, likeCount, publishedAt, updatedAt)
    SELECT c.kind,
           TRY_CAST(c.commentId AS BIGINT)           AS commentId,
           TRY_CAST(c.parentCommentId AS BIGINT)     AS parentCommentId,
           c.channelId,
           TRY_CAST(c.videoId AS BIGINT)             AS videoId,
           c.authorId,
           c.textOriginal,
           TRY_CAST(c.likeCount AS INTEGER)          AS likeCount,
           c.publishedAt,
           c.updatedAt
    FROM temp_comments c
    JOIN videos v ON TRY_CAST(c.videoId AS BIGINT) = v.videoId
    WHERE TRY_CAST(c.commentId AS BIGINT) IS NOT NULL
      AND TRY_CAST(c.commentId AS BIGINT) NOT IN (SELECT commentId FROM comments);
    """)

    conn.execute("DROP TABLE IF EXISTS temp_comments;")
    print(f"Comments import from {comments_parquet_path} completed.")


def main(drop_db=True):
    if drop_db:
        import os
        try:
            os.remove(DB_FILE)
            print(f"Removed existing DB file {DB_FILE}")
        except FileNotFoundError:
            pass

    conn = duckdb.connect(DB_FILE)
    setup_database(conn)

    videos_parquet = DATASET_DIR / "videos.parquet"
    if not videos_parquet.exists():
        print(
            f"ERROR: expected {videos_parquet} to exist. Run the cleaning script first.")
        sys.exit(1)

    import_videos(conn, str(videos_parquet))

    comment_files = sorted(DATASET_DIR.glob("comments*.parquet"))
    if not comment_files:
        print("No comment parquet files found (pattern: comments*.parquet). Nothing more to do.")
    else:
        for cf in comment_files:
            import_comments(conn, str(cf))

    counts = conn.execute("""
    SELECT
      (SELECT COUNT(*) FROM videos) AS videos_count,
      (SELECT COUNT(*) FROM tags) AS tags_count,
      (SELECT COUNT(*) FROM video_tags) AS video_tags_count,
      (SELECT COUNT(*) FROM topic_categories) AS categories_count,
      (SELECT COUNT(*) FROM video_categories) AS video_categories_count,
      (SELECT COUNT(*) FROM comments) AS comments_count,
      (SELECT COUNT(*) FROM orphan_comments) AS orphan_comments_count
    """).fetchone()

    if counts:
        print("Import summary:")
        print(
            f"  videos: {counts[0]}, tags: {counts[1]}, video_tags: {counts[2]}")
        print(f"  categories: {counts[3]}, video_categories: {counts[4]}")
        print(f"  comments: {counts[5]}, orphan_comments: {counts[6]}")

    conn.close()


if __name__ == "__main__":
    main()
