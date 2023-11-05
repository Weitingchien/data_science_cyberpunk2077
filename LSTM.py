import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator



def convert_bools(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_bools(value)
        elif isinstance(value, bool):
            d[key] = True if value else False
    # print(f'd: {d}')
    return d


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



if __name__ == "__main__":
    filename = "2020_11_14_reviews.json"
    df_filtered = pre_processing(filename)
    # 假設 'reviews' 是一個包含所有評論資料的列表
    # 首先將數據轉換為一個pandas DataFrame
    df = pd.DataFrame(df_filtered['review'])  # 轉換為DataFrame

    # 提取時間戳和評論數量
    df['timestamp'] = pd.to_datetime(df_filtered['timestamp_created'], unit='s')
    df.set_index('timestamp', inplace=True)
    df['count'] = 1

    # 按照時間進行重採樣，以日為單位統計評論數量
    daily_reviews = df['count'].resample('D').sum().fillna(0)

    # 數據標準化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(daily_reviews.values.reshape(-1,1))

    # 切分數據集
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

    # 使用TimeseriesGenerator生成時間序列數據
    n_input = 7  # 使用一週的數據來預測下一天
    n_features = 1  # 特徵數為1，因為我們只有評論數量

    generator = TimeseriesGenerator(train_data, train_data, length=n_input, batch_size=1)

    # 建立LSTM模型
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 訓練模型
    model.fit(generator, epochs=5)  # 實際應用中可能需要更多的epochs

    # 使用最後n_input天的數據來進行預測
    x_input = test_data[-n_input:].reshape((1, n_input, n_features))
    predicted_count = model.predict(x_input, verbose=0)

    # 反標準化預測結果
    predicted_count = scaler.inverse_transform(predicted_count)

    print(f'Predicted count: {predicted_count}')



    
