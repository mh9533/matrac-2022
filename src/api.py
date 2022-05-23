import pandas as pd 
import requests

import pathlib 
import json
import datetime
import itertools


class WiseAliceAPI:
    API_URL = "<TODO>"

    def __init__(self, credentials=None, credentials_path="secrets.txt"):
        if credentials:
            email, password = credentials
        else:
            with open(credentials_path, 'rt', encoding="utf8") as f:
                email, password = f.readline().strip().split()
        
        token = self.get_token(email, password)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}"
        })

    def get_token(self, email, password):
        url = self.API_URL + "get_token"
        params = { "email": email, "password": password }
        r = requests.post(url, json=params) 
        return r.json()["data"]["token"]

    def get_devices(self):
        url = self.API_URL + "devices"
        return self.session.get(url).json()

    def get_entry_types(self):
        url = self.API_URL + "entry_types"
        return self.session.get(url).json()

    def get_first_entry_date(self, uuid):
        url = self.API_URL + "first_entry_date"
        return self.session.get(url, params={"uuid": uuid}).json()["data"]

    def get_last_entry_date(self, uuid):
        url = self.API_URL + "last_entry_date"
        return self.session.get(url, params={"uuid": uuid}).json()["data"]

    def get_entries(self, uuid, date):
        """date: YYYY-MM-DD format"""
        url = self.API_URL + "entries"
        return self.session.get(url, params={"date": date, "uuid": uuid}).json()["data"]


def get_entries(api, uuid, start_date=None, end_date=None):
    if start_date is None:
        start_date = api.get_first_entry_date(uuid)
    if end_date is None:
        end_date = api.get_last_entry_date(uuid)  # API can return dates in the future
    date = datetime.date.fromisoformat(start_date)
    end_date = max(datetime.date.today(), datetime.date.fromisoformat(end_date))
    entries = {}
    while date <= end_date:
        print(date.isoformat())
        data = api.get_entries(uuid, date.isoformat())
        entries[date.isoformat()] = data 
        date += datetime.timedelta(days=1)
    return entries

def entries_to_dataframe(entries):
    df = pd.DataFrame(itertools.chain(*entries.values()))
    df['inserted_at'] = pd.to_datetime(df['inserted_at'])
    df = df.sort_values(by=['inserted_at'])
    return df 

def write_entries(entries, path):
    df = entries_to_dataframe(entries)
    df.to_csv(path, index=False)

def read_entries(path):
    df = pd.read_csv(path, parse_dates=['inserted_at'])
    return df


def _download_v1(api, ids, date_range, update=True, dump=False):
    import v1.config as cfg

    uuids = [d["uuid"] for d in api.get_devices() if d["id"] in ids]
    for id_, uuid in zip(ids, uuids):
        path = cfg.get_api_csv_path(id_)
        path.parent.mkdir(exist_ok=True, parents=True)
        start, end = date_range
        if update:
            if path.exists():
                df = read_entries(path)
                start = df['inserted_at'].max().date().isoformat()  # start from end of last date
        entries = get_entries(api, uuid, start_date=start, end_date=end)
        write_entries(entries, path)
        print(path)

        if dump:
            with path.with_suffix('.json').open('w') as f:
                json.dump(entries, f)

def _download_v0():
    api = WiseAliceAPI()
    start_date = None # "2021-04-01"
    end_date = "2022-03-01"
    masters = ["134225002", "134225010", "134225019"]
    uuids = [d["uuid"] for d in api.get_devices() if d["serial"] in masters]
    for serial, uuid in zip(masters, uuids):
        path = pathlib.Path(__file__).resolve().parent.with_name("data") / f"{serial}.json"
        path.parent.mkdir(exist_ok=True, parents=True)
        entries = get_entries(api, uuid, start_date=start_date, end_date=end_date)
        with path.open('w') as f:
            json.dump(entries, f)
        print(path)
        path = path.with_suffix('.csv')
        write_entries(entries, path)
        print(path)
