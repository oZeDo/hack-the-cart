import gc
import pandas as pd
import numpy as np
from collections import Counter


def compute_paip(items: list) -> int:
    # paip = personal average item popularity
    hits = sum(item_hits for _, item_hits in items)
    return hits / len(items)


def sort_personal_item_connections(a: dict):
    rec_dict = {}
    for key, value in a.items():
        rec_dict[key] = []
        for item, hits in value:
            if item not in rec_dict:
                rec_dict[key].append(item)
    return rec_dict


def get_personal_item_connections(df_) -> [list, list]:
    best_items, other_items = {}, {}

    for user_purchases in df_.iteritems():
        user, items = user_purchases
        average = compute_paip(items)
        best_items[user], other_items[user] = [], []

        for item, item_popularity in items:
            if item_popularity > average:
                best_items[user].append((item, item_popularity))
            else:
                other_items[user].append((item, item_popularity))
    return sort_personal_item_connections(best_items), sort_personal_item_connections(other_items)


def get_general_item_connections(basket: list, all_ic: dict) -> list:
    general_ic = []
    for item in basket:
        if item_connection:= all_ic.get(item, None):
            general_ic += item_connection
    return sorted(general_ic, key=lambda x: x[1], reverse=True)


def get_personal_item_connections_by_popularity(personal_connections: list, popularity: dict) -> list:
    general_ic = []
    for item in personal_connections:
        if item_connection:= popularity.get(item, None):
            general_ic += [(item, item_connection)]
    return sorted(general_ic, key=lambda x: x[1], reverse=True)


def merge_item_recommendations(recs):
    rec_dict = {}
    counter = 0
    for item, hits in recs:
        if item not in rec_dict:
            rec_dict[item] = hits
            counter += 1
    return list(rec_dict.keys())


hist_data = pd.read_csv('hist_data.csv').dropna()
test = pd.read_csv('test.csv')

df = hist_data[['item_id', 'pav_order_id', "buyer_id"]]
df_personal = df.groupby(["buyer_id"])["item_id"].agg(lambda x: Counter(x).most_common())

a, b = get_personal_item_connections(df_personal)
best_df = pd.DataFrame({"buyer_id": a.keys(), "personal_ic": a.values()})
other_df = pd.DataFrame({"buyer_id": b.keys(), "other_perconal_ic": b.values()})

del a, b
gc.collect()

df = (
    hist_data[['item_id', 'pav_order_id']]
    .sort_values(['item_id', 'pav_order_id'])
    .merge(hist_data[['item_id', 'pav_order_id']], how='left', on=['pav_order_id'], suffixes=('', '_left'))
)
df_general = df[df['item_id'] != df['item_id_left']].copy()
df_general1 = df_general.groupby(['item_id'])['item_id_left'].agg(lambda x: Counter(x).most_common(3))

all_gic = {k: v for (k, v) in df_general1.iteritems()}

del df_general1, df_general
gc.collect()

test_df = test.groupby(['pav_order_id', 'buyer_id'])['item_id'].agg([('basket', list)])
tmp = test_df['basket'].map(lambda x: merge_item_recommendations(get_general_item_connections(x, all_gic))).reset_index()
tmp = tmp.rename(columns={'basket': "item_by_item"})

items_popularity = df['item_id'].value_counts().reset_index()
items_popularity = items_popularity.rename(columns={"item_id": "count", "index":"item_id"})
items_popularity = zip(items_popularity["item_id"], items_popularity["count"])
items_popularity = {key: value for (key, value) in items_popularity}
tmp1 = other_df['other_perconal_ic'].map(lambda x: merge_item_recommendations(get_personal_item_connections_by_popularity(x, items_popularity)))
b = list(items_popularity.keys())[:20]


a = test_df.reset_index()
a = pd.merge(a, best_df, how="left", on=['buyer_id'])
a['personal_ic'] = a['personal_ic'].fillna("").apply(list)
a = pd.merge(a, tmp, how="left", on=['buyer_id', 'pav_order_id'])
a['other_perconal_ic'] = tmp1
a['other_perconal_ic'] = a['other_perconal_ic'].fillna("").apply(list)
a["most_popular"] = [b for i in a.index]
a['preds'] = np.nan
a['preds'] = a['preds'].fillna("").apply(list)


for row in a.itertuples():
    # + row.other_perconal_ic + row.most_popular
    recommendations = row.personal_ic + row.item_by_item + row.most_popular
    a.at[row.Index, 'preds'] = sorted(set(recommendations), key=recommendations.index)[:20]

a[['pav_order_id', 'preds']].to_csv('prediction.csv', index=False)
