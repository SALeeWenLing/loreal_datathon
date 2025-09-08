import ast
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
from loreal.ontology import LOREAL_ONTOLOGY, GENERIC_YT
from sentence_transformers import SentenceTransformer, util

# Load multilingual model 
MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=False)
def ast_topic_titles(s: str):
    """Parse YouTube topicCategories string -> ['Cosmetics', 'Hairstyle', ...]"""
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        urls = ast.literal_eval(s)
    except Exception:
        return []
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        part = u.split("/wiki/", 1)[-1] if "/wiki/" in u else u.rsplit("/", 1)[-1]
        title = urllib.parse.unquote(part).replace("_", " ").strip()
        if title:
            out.append(title)
    return out

def score_loreal_topics(text: str, yt_titles: list[str]) -> list[str]:
    """
    Simple ontology scorer that we will have to replace with a proper model later:
    - +1 if any ontology keyword is in text (title/description/tags)
    - +1 if any ontology seed appears in YT topic titles (excluding GENERIC_YT)
    Returns topics with score > 0, else ['Other'].
    TODO: MAKE THIS A PROPER MODEL LATER
    """
    text_l = (text or "").lower()
    yt_clean = [t.lower() for t in yt_titles if t and t not in GENERIC_YT]
    yt_join  = " ".join(yt_clean)

    scores = {}
    for topic, node in LOREAL_ONTOLOGY.items():
        kw = node.get("keywords", []) or []
        seeds = node.get("seeds", []) or []
        score = 0
        if kw and any(k.lower() in text_l for k in kw):
            score += 1
        if seeds and any(s.lower() in yt_join for s in seeds):
            score += 1
        if score > 0:
            scores[topic] = score

    if not scores:
        return ["Other"]

    # sort by score desc, then topic name for stability
    return [t for (t, _) in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]

@st.cache_data(show_spinner=False)
def build_video_topics(videos_df):
    """
    Compute ontology topics ONCE per video from:
      - title + description + tags (lexical)
      - YouTube topicCategories titles (seed match; GENERIC_YT down-weighted by excluding)
    """
    v = videos_df.copy()
    if "topicCategories" not in v.columns:
        v["topicCategories"] = None

    v["topic_titles"] = v["topicCategories"].apply(ast_topic_titles)

    def _labels_for_row(row):
        text = f"{row.get('title','')} {row.get('description','')} {row.get('tags','')}"
        return score_loreal_topics(text, row.get("topic_titles", []))

    v["video_topics"] = v.apply(_labels_for_row, axis=1)
    v["video_topic"]  = v["video_topics"].apply(lambda xs: xs[0] if isinstance(xs, list) and xs else "Other")
    v["business_category_primary"] = v["video_topic"]
    return v[["videoId","video_topics","video_topic","business_category_primary"]]

@st.cache_data(show_spinner=False)
def calc_soe_table(pv):
    pv = pv.copy()
    numer = pv["video_likeCount"] + pv["video_commentCount"] + pv["video_favCount"]
    pv["SoE"] = np.where(
        pv["video_viewCount"] > 0,
        numer / pv["video_viewCount"].astype(float),
        np.nan
    )
    return pv[["videoId","SoE"]]


def truncate_text(s: pd.Series, n=90) -> pd.Series:
    '''Truncates text in a pandas Series to a maximum of n characters so that it does not overflow UI elements.'''
    s = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return s.str.slice(0, n) + s.apply(lambda x: "…" if len(x) > n else "")
