import os
import json


def load_json_file(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError()

    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def dump_json_data(json_data, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
