import json
import datetime
import pandas as pd
import steamreviews



# Sample JSON data structure
"""
data = [
    {
        "reviews": {
            "149388403": {
                "recommendationid": "149388403",
                "author": {
                    "steamid": "76561198038934544",
                    "num_games_owned": 0,
                    "num_reviews": 11,
                    "playtime_forever": 4231,
                    "playtime_last_two_weeks": 60,
                    "playtime_at_review": 4231,
                    "last_played": 1698947816
                },
                "language": "english",
                "review": "great game loving it played through quite a few times have phantom city now so replay #5 coming up",
                "timestamp_created": 1698947707,
                "timestamp_updated": 1698947707,
                "voted_up": true,
                "votes_up": 0,
                "votes_funny": 0,
                "weighted_vote_score": 0,
                "comment_count": 0,
                "steam_purchase": true,
                "received_for_free": false,
                "written_during_early_access": false,
                "hidden_in_steam_china": true,
                "steam_china_location": ""
            },
            "149386645": {
                "recommendationid": "149386645",
                "author": {
                    "steamid": "76561198057507801",
                    "num_games_owned": 0,
                    "num_reviews": 2,
                    "playtime_forever": 6373,
                    "playtime_last_two_weeks": 2614,
                    "playtime_at_review": 6347,
                    "last_played": 1698947496
                },
                "language": "english",
                "review": "Great game to chill and spend time, i recommend not going straight for the main story since doing side jobs will get you tonsof gear and experience. \nThe new DLC added some insane features that are really game changing.",
                "timestamp_created": 1698945927,
                "timestamp_updated": 1698945927,
                "voted_up": true,
                "votes_up": 0,
                "votes_funny": 0,
                "weighted_vote_score": 0,
                "comment_count": 0,
                "steam_purchase": true,
                "received_for_free": false,
                "written_during_early_access": false,
                "hidden_in_steam_china": true,
                "steam_china_location": ""
            },
    }
    }
    # ... more reviews
]
"""





def convert_bools(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_bools(value)
        elif isinstance(value, bool):
            d[key] = True if value else False
    # print(f'd: {d}')
    return d

    # Flatten the nested JSON data



def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        print(f'x: {x}')
        if type(x) is dict:
            for a in x:
                print(f'a: {a}')
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



def main(filename, output_filename):
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
    print(f'flat_list: {flat_list}')

    # Create a DataFrame from the flat list
    df = pd.DataFrame(flat_list)

    # Export the DataFrame to a CSV file
    df.to_csv('reviews.csv', index=False)



def timestamp_to_date(timestamp):
    # timestamp = 1697743335  # 這裡放入你的時間戳
    date = datetime.datetime.fromtimestamp(timestamp)

    print(date)


if __name__ =="__main__":
    timestamp_to_date(1697845125)
    """
    days = 366
    for i in range(0, days, 31):
        print(i)
    """
    # app_id = [1091500]
    """
    request_params = dict()
    review_dict_temp = []
    # Reference: https://partner.steamgames.com/doc/store/getreviews
    request_params['language'] = 'english'
    # request_params['review_type'] = 'positive'
    request_params['purchase_type'] = 'steam'
    request_params['filter'] = 'all'  # reviews are sorted by helpfulness instead of chronology
    request_params['day_range'] = "1095"  # focus on reviews which were published during the past four weeks
    review_dict, query_count = steamreviews.download_reviews_for_app_id(1091500,
                                                                    chosen_request_params=request_params)

    print(f'review_dict: {review_dict}')
    review_dict_temp.append(review_dict)
    filename = 'reviews.json'

    print(f'review_dict_temp: {len(review_dict_temp)}')

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(review_dict_temp, f, ensure_ascii=False, indent=4)

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
    df.to_csv(filename, index=False)
    """
    
    """
    
    days = 365
    for i in range(days):
        print(i)
    """
    # main(filename, output_filename)