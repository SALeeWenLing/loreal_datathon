import duckdb
import pandas as pd
from langdetect import detect_langs
from tqdm import tqdm
import argparse
import time
import re
from collections import Counter

# Config
DB_FILE = "loreal_datathon.duckdb"
VIDEO_BATCH = 1000
NUM_COMMENT_SAMPLE = 20
CONF_THRESHOLD = 0.60

# Weights
W_LANG = 1.0
W_LANG_COMMENTS = 0.6

# Language to country mapping
lang_map = {
    "ms": ["MY"], "id": ["ID"],
    "en": ["US", "GB", "CA", "AU", "NZ", "IE", "ZA", "SG", "IN"],
    "th": ["TH"], "vi": ["VN"], "tl": ["PH"],
    "fr": ["FR", "CA", "BE", "CH", "LU", "MC"],
    "es": ["ES", "MX", "AR", "CO", "CL", "PE", "VE", "EC", "GT", "CU", "DO", "HN", "PY", "SV", "NI", "CR", "PA", "UY", "BO"],
    "pt": ["PT", "BR", "AO", "MZ"],
    "de": ["DE", "AT", "CH", "LU", "BE", "LI"],
    "zh-cn": ["CN"], "zh-tw": ["TW"], "zh": ["CN", "TW", "HK", "SG", "MY"],
    "ja": ["JP"], "ko": ["KR", "KP"],
    "ru": ["RU", "BY", "KZ", "KG", "UA"],
    "ar": ["EG", "SA", "DZ", "IQ", "MA", "SD", "YE", "SY", "TN", "JO", "AE", "LY", "LB", "OM", "KW", "MR", "QA", "BH"],
    "hi": ["IN"], "bn": ["BD", "IN"], "pa": ["IN", "PK"], "te": ["IN"], "mr": ["IN"], "ta": ["IN", "LK", "SG"],
    "ur": ["PK", "IN"], "fa": ["IR", "AF", "TJ"], "tr": ["TR", "CY"],
    "it": ["IT", "CH", "SM", "VA"], "nl": ["NL", "BE", "SR"],
    "pl": ["PL"], "uk": ["UA"], "ro": ["RO", "MD"], "hu": ["HU"],
    "sv": ["SE", "FI"], "da": ["DK"], "no": ["NO"], "fi": ["FI"],
    "cs": ["CZ"], "sk": ["SK"], "bg": ["BG"], "el": ["GR", "CY"],
    "he": ["IL"], "ps": ["AF"], "ku": ["IQ", "TR", "IR", "SY"],
    "sw": ["KE", "TZ", "UG"], "am": ["ET"], "ha": ["NG", "NE"],
    "yo": ["NG"], "ig": ["NG"], "zu": ["ZA"], "xh": ["ZA"], "af": ["ZA"],
    "ne": ["NP"], "si": ["LK"], "my": ["MM"], "km": ["KH"], "lo": ["LA"],
    "ka": ["GE"], "hy": ["AM"], "az": ["AZ"], "uz": ["UZ"], "kk": ["KZ"],
    "ky": ["KG"], "tg": ["TJ"], "tk": ["TM"], "mn": ["MN"],
    "ca": ["ES", "AD"],  # Catalan for Spain and Andorra
    "eu": ["ES"],  # Basque (Spain)
    "gl": ["ES"],  # Galician (Spain)
    "cy": ["GB"],  # Welsh (UK)
    "ga": ["IE"],  # Irish (Ireland)
    "gd": ["GB"],  # Scottish Gaelic (UK)
    "br": ["FR"],  # Breton (France)
    "fy": ["NL"],  # West Frisian (Netherlands)
    "lb": ["LU"],  # Luxembourgish (Luxembourg)
    "mt": ["MT"],  # Maltese (Malta)
    "is": ["IS"],  # Icelandic (Iceland)
    "fo": ["FO"],  # Faroese (Faroe Islands)
    "sm": ["WS"],  # Samoan (Samoa)
    "mi": ["NZ"],  # Maori (New Zealand)
    "haw": ["US"],  # Hawaiian (USA)
}

# Country frequency weights for fashion/beauty context
country_frequency_weights = {
    'US': 1.2, 'GB': 1.1, 'CA': 1.1, 'AU': 1.1, 'DE': 1.05, 'FR': 1.05,
    'BR': 1.05, 'IN': 1.05, 'RU': 1.05, 'JP': 1.05, 'KR': 1.05, 'CN': 1.05,
    'ES': 1.03, 'IT': 1.03, 'MX': 1.03, 'ID': 1.03, 'NL': 1.02, 'SE': 1.02,
    # Fashion/beauty hubs get a slight boost
    'FR': 1.15, 'IT': 1.1, 'US': 1.1, 'GB': 1.08, 'JP': 1.08, 'KR': 1.08,
    # Reduce weight for unlikely fashion/beauty hubs
    'AD': 0.1, 'LI': 0.1, 'MC': 0.1, 'SM': 0.1, 'VA': 0.05,
}

# Pre-compile regex for faster text cleaning
CLEAN_TEXT = re.compile(r'[^\w\s]')

# Common English words for verification (expanded list)
ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
    'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
    'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
    'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
    'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
    # Fashion/beauty specific words
    'makeup', 'beauty', 'skincare', 'fashion', 'style', 'cosmetic', 'product',
    'review', 'tutorial', 'hair', 'skin', 'care', 'lipstick', 'mascara', 'foundation',
    'blush', 'eyeshadow', 'liner', 'primer', 'concealer', 'powder', 'highlighter',
    'contour', 'brush', 'sponge', 'blend', 'apply', 'routine', 'morning', 'night',
    'cleanse', 'moisturize', 'serum', 'toner', 'mask', 'sunscreen', 'protection',
    'glow', 'shine', 'matte', 'dewy', 'natural', 'glam', 'everyday', 'special',
    'occasion', 'party', 'wedding', 'event', 'professional', 'expert', 'tips',
    'tricks', 'hacks', 'drugstore', 'luxury', 'affordable', 'expensive', 'quality',
    'brand', 'company', 'loreal', 'maybelline', 'estee', 'lauder', 'mac', 'nars',
    'clinique', 'dior', 'chanel', 'ysl', 'gucci', 'prada', 'versace', 'hudabeauty',
    'jaclynhill', 'jeffreestar', 'tati', 'jamescharles', 'nikketutorials'
}

# Languages that are often confused with English
CONFUSED_LANGUAGES = {'da', 'ca', 'no', 'af', 'sv', 'nl', 'de'}

# Characteristic words for confused languages (to help distinguish)
LANGUAGE_INDICATORS = {
    'da': ['og', 'i', 'det', 'at', 'en', 'et', 'som', 'på', 'for', 'der', 'til'],
    'ca': ['i', 'el', 'de', 'que', 'la', 'els', 'en', 'un', 'una', 'és', 'per'],
    'no': ['og', 'i', 'er', 'det', 'at', 'en', 'et', 'som', 'på', 'for', 'å'],
    'af': ['en', 'die', 'het', 'in', 'nie', 'sy', 'vir', 'om', 'op', 'uit', 'met'],
    'sv': ['och', 'i', 'att', 'en', 'ett', 'som', 'på', 'för', 'av', 'med', 'är'],
    'nl': ['en', 'de', 'het', 'in', 'van', 'op', 'ik', 'te', 'dat', 'die', 'voor'],
    'de': ['und', 'ich', 'die', 'der', 'das', 'in', 'den', 'von', 'mit', 'sich', 'für']
}


def safe_str(value):
    """Safely convert value to string, handling None and NaN"""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def is_likely_english(text, min_english_ratio=0.25):
    """Check if text is likely English based on common words"""
    if not text or len(text.strip()) < 10:
        return False

    words = text.lower().split()
    if len(words) < 5:  # Too short to reliably determine
        return False

    english_word_count = sum(1 for word in words if word in ENGLISH_WORDS)
    english_ratio = english_word_count / len(words)

    return english_ratio >= min_english_ratio


def has_language_indicators(text, language_code):
    """Check if text contains characteristic words of a specific language"""
    if not text or language_code not in LANGUAGE_INDICATORS:
        return False

    text_lower = text.lower()
    indicators = LANGUAGE_INDICATORS[language_code]

    # Count how many characteristic words appear
    indicator_count = sum(1 for word in indicators if word in text_lower)

    # If we find multiple characteristic words, it's likely this language
    return indicator_count >= 2  # At least 2 characteristic words


def detect_language_fast(text):
    """Faster language detection with text length check and English verification"""
    try:
        if not text or len(text.strip()) < 10:
            return None, 0.0

        # Clean text for better language detection
        clean_text = CLEAN_TEXT.sub(' ', text.lower())
        if len(clean_text) < 5:
            return None, 0.0

        probs = detect_langs(clean_text)
        if not probs:
            return None, 0.0

        best = probs[0]

        # If the detected language is one that's often confused with English
        if best.lang in CONFUSED_LANGUAGES:
            # Check if it has strong indicators for that language
            has_strong_indicators = has_language_indicators(
                clean_text, best.lang)

            # Check if it's likely English
            is_english = is_likely_english(clean_text)

            # If it has strong indicators for the confused language, trust the detection
            if has_strong_indicators:
                return best.lang, min(best.prob + 0.1, 1.0)  # Boost confidence
            # If it's likely English, override to English
            elif is_english:
                # Check if English is in the probabilities
                for prob in probs:
                    if prob.lang == 'en':
                        return 'en', max(prob.prob, 0.7)
                return 'en', 0.7  # Default English confidence

        # If English is detected with low confidence, verify with word list
        if best.lang == 'en' and best.prob < 0.8:
            if is_likely_english(clean_text):
                return 'en', 0.8  # Boost confidence for verified English
            else:
                # Check if another language has high confidence
                if len(probs) > 1 and probs[1].prob > 0.7:
                    return probs[1].lang, probs[1].prob

        # If non-English is detected but text looks like English, verify
        if best.lang != 'en' and best.prob < 0.9 and is_likely_english(clean_text):
            # Check if English is in the top probabilities
            for prob in probs:
                if prob.lang == 'en' and prob.prob > 0.1:
                    # Return English with boosted confidence
                    return 'en', max(prob.prob, 0.7)

        # Only return if confidence is above threshold
        if best.prob >= 0.6:
            return best.lang, best.prob
        else:
            return None, 0.0
    except Exception:
        return None, 0.0


def get_language_weight(lang_code, text):
    """Get weight for language based on context"""
    base_weight = W_LANG

    # Reduce weight for languages that are often misclassified
    if lang_code == "ca":  # Catalan
        # Check if text contains indicators of Catalan context
        catalan_indicators = ["català", "catalan",
                              "barcelona", "catalunya", "andorra"]
        text_lower = text.lower()
        if not any(indicator in text_lower for indicator in catalan_indicators):
            return base_weight * 0.3  # Reduce weight if no Catalan context

    return base_weight


def run_inference(limit=None):
    conn = duckdb.connect(DB_FILE)
    conn.execute("""
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS inferred_region VARCHAR DEFAULT NULL;
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS inferred_region_confidence DOUBLE DEFAULT NULL;
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS inferred_region_source VARCHAR DEFAULT NULL;
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS inferred_region_updated_at TIMESTAMP DEFAULT now();
    -- New columns for storing detected languages
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS detected_language VARCHAR DEFAULT NULL;
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS detected_language_confidence DOUBLE DEFAULT NULL;
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS comments_language VARCHAR DEFAULT NULL;
    ALTER TABLE videos ADD COLUMN IF NOT EXISTS comments_language_confidence DOUBLE DEFAULT NULL;
    """)

    # Count total videos
    count_q = "SELECT COUNT(*) FROM videos"
    total_videos = conn.execute(count_q).fetchone()
    if total_videos is None:
        raise Exception("total_videos is None (duckdb query)")
    total_videos = total_videos[0]
    print(f"Total videos in DB: {total_videos}")

    # Pre-cache comments for all videos (one big query)
    print("Pre-caching comments...")
    comments_q = """
    WITH ranked_comments AS (
        SELECT videoId, textOriginal,
               ROW_NUMBER() OVER (PARTITION BY videoId ORDER BY likeCount DESC NULLS LAST) as rn
        FROM comments
    )
    SELECT videoId, STRING_AGG(textOriginal, ' ') as combined_text
    FROM ranked_comments
    WHERE rn <= %d
    GROUP BY videoId
    """ % NUM_COMMENT_SAMPLE
    comments_df = conn.execute(comments_q).fetchdf()
    comments_cache = dict(
        zip(comments_df['videoId'], comments_df['combined_text']))
    print(f"Cached comments for {len(comments_cache)} videos")

    # Fetch videos in batches
    offset = 0
    processed = 0

    # Precompute total batches for progress tracking
    total_batches = (total_videos + VIDEO_BATCH - 1) // VIDEO_BATCH
    if limit:
        total_batches = min(
            total_batches, (limit + VIDEO_BATCH - 1) // VIDEO_BATCH)

    pbar = tqdm(total=total_videos if not limit else min(limit, total_videos))

    while True:
        # Fetch a batch of videos
        q = f"""
        SELECT videoId, title, description
        FROM videos
        ORDER BY videoId
        LIMIT {VIDEO_BATCH} OFFSET {offset}
        """
        df = conn.execute(q).fetchdf()
        if df.empty:
            break

        # Process each video in the batch
        results = []
        for _, row in df.iterrows():
            vid = int(row['videoId'])
            title = safe_str(row['title'])
            desc = safe_str(row['description'])
            comments_text = safe_str(comments_cache.get(vid, ""))

            # Build signals - now only language-based
            votes = {}

            # 1) Language from title + description
            combined_text = " ".join([title, desc])
            lang, lang_conf = detect_language_fast(
                combined_text) if combined_text and combined_text.strip() else (None, 0.0)

            # 2) Language from comments
            comments_lang, comments_lang_conf = detect_language_fast(
                comments_text) if comments_text and comments_text.strip() else (None, 0.0)

            def lang_vote(lang_code, conf, weight, context_text=""):
                if not lang_code:
                    return
                candidates = lang_map.get(lang_code)
                if not candidates:
                    return

                # Get context-adjusted weight
                adjusted_weight = get_language_weight(
                    lang_code, context_text) * conf

                # Vote for all candidates with distributed weight
                vote_weight = adjusted_weight / len(candidates)
                for candidate in candidates:
                    # Apply country frequency weighting
                    freq_weight = country_frequency_weights.get(candidate, 1.0)
                    votes[candidate] = votes.get(
                        candidate, 0.0) + vote_weight * freq_weight

            # Vote based on title+description language
            lang_vote(lang, lang_conf, W_LANG, combined_text)

            # Vote based on comments language (with lower weight)
            lang_vote(comments_lang, comments_lang_conf,
                      W_LANG_COMMENTS, comments_text)

            # If we have no votes, try to use the most common language from the batch as fallback
            if not votes:
                # Fallback: use the most common language in the batch if available
                if lang:
                    lang_vote(lang, 0.5, W_LANG * 0.3, combined_text)
                elif comments_lang:
                    lang_vote(comments_lang, 0.5,
                              W_LANG_COMMENTS * 0.3, comments_text)

            # Aggregate votes with minimum threshold
            if votes:
                # Filter out countries with very low votes
                filtered_votes = {k: v for k, v in votes.items() if v >= 0.3}

                if filtered_votes:
                    total_score = sum(filtered_votes.values())
                    best_country, best_score = max(
                        filtered_votes.items(), key=lambda x: x[1])
                    confidence = best_score / total_score if total_score > 0 else 0.0
                else:
                    # If no country meets the threshold, use the one with the highest votes anyway
                    best_country, best_score = max(
                        votes.items(), key=lambda x: x[1])
                    confidence = best_score / \
                        sum(votes.values()) if sum(votes.values()) > 0 else 0.0
            else:
                best_country, confidence = None, 0.0

            # Prepare source info
            source_parts = []
            if lang:
                source_parts.append(f"lang={lang}({lang_conf:.2f})")
            if comments_lang:
                source_parts.append(
                    f"comments_lang={comments_lang}({comments_lang_conf:.2f})")
            source = ",".join(source_parts) if source_parts else "none"

            src_tag = "inferred" if confidence >= CONF_THRESHOLD else "low_confidence"

            # Store both region inference and language information
            results.append((
                best_country,
                float(confidence),
                src_tag + ":" + source,
                lang,  # detected_language
                # detected_language_confidence
                float(lang_conf) if lang_conf else None,
                comments_lang,  # comments_language
                # comments_language_confidence
                float(comments_lang_conf) if comments_lang_conf else None,
                int(vid)
            ))

            pbar.update(1)

        # Update database with all results
        if results:
            conn.executemany("""
            UPDATE videos
            SET inferred_region = ?,
                inferred_region_confidence = ?,
                inferred_region_source = ?,
                detected_language = ?,
                detected_language_confidence = ?,
                comments_language = ?,
                comments_language_confidence = ?,
                inferred_region_updated_at = now()
            WHERE videoId = ?;
            """, results)

        processed += len(df)
        offset += VIDEO_BATCH
        print(f"Processed {processed} videos so far...")

        if limit and processed >= limit:
            break

    pbar.close()
    conn.close()
    print("Finished inference pass. Processed:", processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="limit number of videos to process")
    args = parser.parse_args()

    start_time = time.time()
    run_inference(limit=args.limit)
    end_time = time.time()

    print(f"Total processing time: {end_time - start_time:.2f} seconds")
