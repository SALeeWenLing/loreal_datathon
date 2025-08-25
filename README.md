# L'Oréal CommentSense: Topic & Engagement Analysis

This Streamlit app helps you analyze comment data for L'Oréal using topic modeling (BERTopic) and engagement metrics (likes, replies).

## Features
- Upload a CSV file of comments with sharing metrics
- Extract topics from comments using BERTopic
- Aggregate and visualize likes and replies by topic
- Interactive dashboard for business insights

## Getting Started

### 1. Clone the repository
```sh
git clone https://github.com/SALeeWenLing/loreal_datathon.git
cd loreal_datathon
```

### 2. Set up a virtual environment
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Run the app
```sh
streamlit run app.py
```

### 5. Upload your data
- Use the provided `comments_sample.csv` or upload your own CSV with columns:
	- `comment_text` (text)
	- `num_replies` (int)
	- `num_likes` (int)

## Example CSV
```
comment_text,num_replies,num_likes
"I love this new shampoo! My hair feels amazing.",5,23
"The packaging is so pretty, but the scent is too strong.",2,10
...
```

## Notes
- The app uses BERTopic for topic modeling. The first run may take a few minutes to download models.


# loreal_datathon

