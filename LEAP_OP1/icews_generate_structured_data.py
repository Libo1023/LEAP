import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter

def main(project_path, country):
    print('------generate structured data for country {}'.format(country))

    data_path = project_path + '/data'
    data = pd.read_csv("{}/{}/train.txt".format(data_path, country), sep = "\t", 
                       names = ["subject", "relation", "object", "timestamp"])
    print(data.shape)
    # print(data)
    s_count = data["subject"].to_numpy()
    o_count = data["object"].to_numpy()
    print(s_count.shape, o_count.shape)
    ents_train = np.hstack((s_count, o_count))
    list_ents_train = ents_train.tolist()
    print(ents_train.shape, len(list_ents_train))
    
    train_stat_ent = Counter(list_ents_train)
    freq_head_ents = sum(train_stat_ent.values()) // 3
    train_stat_ent = sorted(train_stat_ent.items(), key=lambda item: item[1], reverse=True)
    head_ents = []
    total_freq_ent = 0
    for entity_id, freq in train_stat_ent:
        if total_freq_ent <= freq_head_ents:
            head_ents.append(entity_id)
            total_freq_ent += freq
        else:
            break

    output_path = '{}/{}'.format(data_path, country)

    json.dump(head_ents, open(output_path + '/head_ents.json', 'w'), indent=4)
    print("head_ents.json file saved!")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate structured data')
    parser.add_argument("--project_path", type=str, default="..",
                        help="project root path")
    parser.add_argument("--c", type=str, default="AFG",
                        help="country: AFG, IND, IRN, NGA, or RUS.")

    args = parser.parse_args()

    main(args.project_path, args.c)

