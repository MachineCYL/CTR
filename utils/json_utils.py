# -*- coding: utf-8 -*-
import json


def save_json(file_path, data):
    with open(file_path, mode="w", encoding="utf-8") as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)


def read_json(file_path):
    with open(file=file_path, mode="r", encoding="utf-8") as fr:
        data = json.load(fr)
    return data
