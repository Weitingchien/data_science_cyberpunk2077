import json
import torch
import steamreviews
import pandas as pd
import torch.nn as nn
from time import sleep
from textblob import TextBlob #用於情感分析
from prophet import Prophet #用於時間序列預測
from transformers import BertTokenizer
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader, random_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# 繪製學習曲線
def plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = list(range(1, len(train_losses)+1))  # 更新這裡，使用train_losses的長度作為epochs數量
    plt.figure(figsize=(12, 6))

    # 繪製損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)  # 設定x軸的刻度為整數的epoch值
    plt.legend()

    # 繪製準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)  # 同樣設定x軸的刻度
    plt.legend()

    plt.tight_layout()
    plt.show()







# 評估模型在驗證集上的表現
def evaluate_model_on_val_set(val_loader, model, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, sentiment = batch
            text_lengths = torch.sum(attention_mask, dim=1)
            predictions = model(input_ids, text_lengths)
            loss = criterion(predictions, sentiment)
            _, predicted = torch.max(predictions.data, 1)
            correct_predictions += (predicted == sentiment).sum().item()
            total_samples += sentiment.size(0)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy









def evaluate_model_on_test_set(test_loader, model, criterion, optimizer):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, sentiment = batch
            text_lengths = torch.sum(attention_mask, dim=1)

            predictions = model(input_ids, text_lengths)

            loss = criterion(predictions, sentiment)
            _, predicted = torch.max(predictions.data, 1)

            # 用於紀錄正確預測的樣本數
            correct_predictions += (predicted == sentiment).sum().item()
            total_samples += sentiment.size(0)
            # 累積當前批次的損失
            total_loss += loss.item()

        # 計算平均損失與準確率
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_samples
        print(f'Test Loss: {avg_loss:.4f} ')
        print(f'Test Accuracy: {accuracy:.4f} ')






def train_model_with_early_stopping(train_loader, val_loader, model, criterion, optimizer, num_epochs, patience=3):
    train_losses, val_losses = [], [] # 用於紀錄每個epoch的訓練與驗證損失
    train_accuracies, val_accuracies = [], []  # 用於紀錄每個epoch的訓練與驗證準確率
    best_val_loss = float('inf') # 初始化最佳驗證損失為無限大(inf)
    epochs_no_improve = 0 # 初始化連續沒有改善的epoch數為0
    best_model_state = None # 初始化最佳模型參數狀態為None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            input_ids, attention_mask, sentiment = batch
            text_lengths = torch.sum(attention_mask, dim=1) # 使用BERT的attention mask來計算文本長度
            predictions = model(input_ids, text_lengths)
            loss = criterion(predictions, sentiment) # 計算預測與實際目標之間的損失
            _, predicted = torch.max(predictions.data, 1)
            correct_predictions += (predicted == sentiment).sum().item()
            total_samples += sentiment.size(0)
            optimizer.zero_grad() # 梯度歸零
            loss.backward() # 反向傳播, 計算梯度
            optimizer.step() # 根據計算的梯度更新模型參數
            total_loss += loss.item() # 累加當前批次的損失

        avg_train_loss = total_loss / len(train_loader) # 計算平均訓練損失
        train_accuracy = correct_predictions / total_samples # 計算訓練準確率
        train_losses.append(avg_train_loss) # 將當前epoch的訓練損失加入list中
        train_accuracies.append(train_accuracy) # 將當前epoch的訓練準確率加入list中

        val_loss, val_accuracy = evaluate_model_on_val_set(val_loader, model, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # 早停法檢查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # 儲存當前最佳模型的參數
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping triggered')
                break
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model_state.pth') # 儲存最佳模型參數
        print('Best model saved to best_model.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies








class ReviewsDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        review = self.df.iloc[idx, self.df.columns.get_loc('review')]
          # 確保review是字串
        if not isinstance(review, str):
            review = str(review) if review is not None else ""
        # print(f'review: {review}')
        tokens = self.tokenizer(review, max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        # torch.long: 64位整數
        sentiment = torch.tensor(self.df.iloc[idx, self.df.columns.get_loc('sentiment_label')], dtype=torch.long)

        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        # print(f'input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}, sentiment: {sentiment}')
        return input_ids, attention_mask, sentiment




class SentimentAnalysis_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)












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
    # request_params['purchase_type'] = 'all'
    request_params['filter'] = 'all'
    request_params['review_type'] = 'all'
    # request_params['filter'] = 'recent'
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
        df_filtered['sentiment_score'] = df_filtered['review'].apply(lambda x: sia.polarity_scores(x)['compound'])
        df_filtered['sentiment_label'] = df_filtered['sentiment_score'].apply(sentiment_label)
        df_filtered['date'] = pd.to_datetime(df_filtered['timestamp_created'], unit='s')
        print('df_filtered: ', df_filtered)
        df_filtered.to_csv(filename, index=False)
    except Exception as e:
        print(f"無法將資料寫入CSV檔案: {e}")
    return df_filtered



def sentiment_label(score):
    if score <= -0.05:
        return 0  # 負面
    elif score >= 0.05:
        return 1  # 正面
    else:
        return 2  # 中性



def sentiment_score(review):
    score = sia.polarity_scores(review)['compound']
    return 1 if score > 0.1 else 0


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 512  # 可以根據需要調整
    batch_size = 32
    dataset = ReviewsDataset(csv_file='2020_12_05_reviews.csv', tokenizer=tokenizer, max_len=max_len)
    # 訓練集,測試集,驗證集的大小(70%,10%,20%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    print(f'test_size: {test_size}')

    # 隨機切分數據集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 創建對應的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    
    for i, batch in enumerate(train_loader):
        print(f'第{i}次迭代')
        input_ids, attention_mask, sentiment = batch
        print(f'input_ids: {input_ids}')
        print(f'attention_mask: {attention_mask}')
        print(f'sentiment: {sentiment}')
        if i==0:
            break

    # 假設您已經有 train_loader 和 test_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = len(tokenizer.vocab)  # 根據您的分詞器
    dropout = 0.5   # Dropout 比率
    n_layers = 2
    hidden_dim = 256
    output_dim = 3  # 負面、中性、正面
    embedding_dim = 768
    bidirectional = False
    num_epochs = 10
    

    model = SentimentAnalysis_LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
    # model.parameters() 會返回模型中所有可訓練的參數
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_early_stopping(train_loader, val_loader, model, criterion, optimizer, num_epochs)
    plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    model_state = torch.load('best_model_state.pth')
    model.load_state_dict(model_state)
    evaluate_model_on_test_set(test_loader, model, criterion, optimizer)





if __name__ == "__main__":
    """
    days = 1095
    for i in range(31, days, 31):
        days_str, filename = dataloader(i)
        pre_processing(filename)
        sleep(20)
    """
    """
    days_str, filename = dataloader(1095)
    pre_processing(filename)
    """
    
    filename = "2020_12_05_reviews.json"
    df_filtered = pre_processing(filename)
    main()
    
    
    
