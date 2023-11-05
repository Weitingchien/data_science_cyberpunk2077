import json
from time import sleep
import pandas as pd
import steamreviews
from textblob import TextBlob #用於情感分析
from prophet import Prophet #用於時間序列預測
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt



sia = SentimentIntensityAnalyzer()

def dataloader(days):
    print(f'days: {days}')
    # 現在的日期和時間
    now = datetime.now()
    # 計算_天之前的日期
    _days_ago = now - timedelta(days=days)
    timestamp = int(_days_ago.timestamp())
    # 只取年月日的部分格式化日期
    formatted_date = _days_ago.strftime('%Y_%m_%d')
    # 輸出看看
    print("_days_ago", _days_ago)
    print("timestamp:", timestamp)
    print("formatted_date:", formatted_date)

    request_params = dict()
    # Reference: https://partner.steamgames.com/doc/store/localization#supported_languages
    request_params['language'] = 'english'
    # Reference: https://partner.steamgames.com/doc/store/getreviews
    # request_params['review_type'] = 'positive'
    request_params['purchase_type'] = 'steam'
    request_params['filter'] = 'all'
    days_str = str(days)
    print(f'days_str: {type(days_str)}')
    request_params["day_range"] = days_str

    review_dict_temp = []


    print(f'request_params["day_range"]: {request_params["day_range"]}')
    app_id = 1091500
    review_dict, query_count = steamreviews.download_reviews_for_app_id(app_id,
                                                                    chosen_request_params=request_params)
    
    review_dict_temp.append(review_dict)
    filename = f'{formatted_date}_reviews.json'
    print(f'review_dict_temp: {len(review_dict_temp)}')

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(review_dict_temp, f, ensure_ascii=False, indent=4)

    print(f'review_dict_temp: {len(review_dict_temp)}')
    return days_str, filename

"""
def convert_to_csv():
    with open('reviews.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.to_csv('reviews.csv', index=False)
"""

def convert_bools(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_bools(value)
        elif isinstance(value, bool):
            d[key] = True if value else False
    # print(f'd: {d}')
    return d

# 巢狀json轉成平面json
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out




def pre_processing(filename):

    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        convert_bools(item)


    # Process each review and store it in a list
    flat_list = []
    for entry in data:
        for review_id, review in entry["reviews"].items():
            flat_review = flatten_json(review)
            flat_review['review_id'] = review_id  # Add the review_id to the flat dictionary
            flat_list.append(flat_review)
    # print(f'flat_list: {flat_list}')

    # Create a DataFrame from the flat list
    df = pd.DataFrame(flat_list)
    total_rows = df.shape[0]
    print("總筆數(過濾前):", total_rows)
    # 匹配中文、英文、數字、某些標點符號。
    pattern_valid = r'[\u4e00-\u9fffA-Za-z0-9\s,.!?\'\"]+'
    # 捕獲重複四次以上的非中英文數字或標點的字符（可能的ASCII藝術）。
    pattern_repeat = r'([^\u4e00-\u9fffA-Za-z0-9\s,.!?\'\"])\1{3,}'

    # 先篩選出含有有效字符的列。
    df_valid = df[df['review'].str.contains(pattern_valid, regex=True, na=False)]

    # 然後在這些列中過濾掉含有重複字符的列。
    df_filtered = df_valid[~df_valid['review'].str.contains(pattern_repeat, regex=True, na=False)]
    total_rows = df_filtered.shape[0]
    print("總筆數(過濾後):", total_rows)
    # Export the DataFrame to a CSV file
    filename = filename.split('.')[0] + '.csv'
    print(f'filename: {filename}')
    print(df_filtered.columns)
    df_filtered = df_filtered.drop(['steam_china_location', 'written_during_early_access', 'received_for_free', 'weighted_vote_score', 'votes_funny', 'votes_up', 'author_playtime_last_two_weeks'], axis=1)
    try:
        df_filtered.to_csv(filename, index=False)
    except Exception as e:
        print(f"無法將資料寫入CSV檔案: {e}")
    return df_filtered



def sentiment_score(review):
    return sia.polarity_scores(review)['compound']


def main(df_filtered):
    df_filtered['sentiment'] = df_filtered['review'].apply(sentiment_score)
    df_filtered['date'] = pd.to_datetime(df_filtered['timestamp_created'], unit='s')

    # df_filtered.to_csv("2022_11_27_reviews.csv", index=False)
    #  聚集每日的評論數據來觀察趨勢
    daily_sentiment = df_filtered.resample('D', on='date')['sentiment'].mean()
    # daily_sentiment.to_csv("2022_11_27_reviews.csv", index=False)
    print(daily_sentiment)

    
    plt.figure(figsize=(10,5))
    plt.plot(daily_sentiment.index, daily_sentiment, marker='o')
    plt.title('Daily Average Sentiment Score for Cyberpunk 2077 Reviews')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.show()
    



if __name__ == "__main__":
    """
    days = 1095
    for i in range(0, days, 31):
        days_str, filename = dataloader(i)
        pre_processing(filename)
        sleep(20)
    """
    
    filename = "2020_11_14_reviews.json"
    df_filtered = pre_processing(filename)
    main(df_filtered)
    
