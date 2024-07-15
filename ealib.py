
import requests
import logging
import pandas as pd
from typing import List

"""
COMPANY TICKERS dataframe getter
"""
def get_tickers_df(headers) -> pd.DataFrame:
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(tickers_url, headers=headers)

    resp_code = response.status_code 
    if resp_code < 200 or resp_code >= 300:
        logging.warning(f"Request to {tickers_url} was unsuccessful. Response code: {resp_code}")
        return None
    logging.info(f"Request to {tickers_url} returned successfully. Response code: {resp_code}")

    tickers_dict = response.json()
    tickers_df = pd.DataFrame.from_dict(tickers_dict, orient="index")
    # Pad with leading zeroes the cik, in order to make them 10-digit
    tickers_df["cik_str"] = tickers_df["cik_str"].astype(str).str.zfill(10)
    return tickers_df

"""
COMPANY-FILING-METADATA dict getter
"""
def get_metadata_dict(headers, cik) -> dict:
    metadata_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(metadata_url, headers=headers)

    resp_code = response.status_code 
    if resp_code < 200 or resp_code >= 300:
        logging.warning(f"Request to {metadata_url} was unsuccessful. Response code: {resp_code}")
        return None
    logging.info(f"Request to {metadata_url} returned successfully. Response code: {resp_code}")

    metadata_dict = response.json()
    return metadata_dict

"""
COMPANY-FACTS-DATA dict getter
"""
def get_companyfacts_dict(headers, cik) -> dict:
    companyfacts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(companyfacts_url, headers=headers)

    resp_code = response.status_code 
    if resp_code < 200 or resp_code >= 300:
        logging.warning(f"Request to {companyfacts_url} was unsuccessful. Response code: {resp_code}")
        return None
    logging.info(f"Request to {companyfacts_url} returned successfully. Response code: {resp_code}")

    companyfacts_dict = response.json()
    return companyfacts_dict

"""
COMPANY-CONCEPT dict getter
    e.g. concept = /us-gaap/Revenues
"""
def get_companyconcept_dict(headers, cik, concept) -> dict:
    cc_url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}'f'{concept}.json'
    response = requests.get(cc_url, headers=headers)

    resp_code = response.status_code 
    if resp_code < 200 or resp_code >= 300:
        logging.warning(f"Request to {cc_url} was unsuccessful. Response code: {resp_code}")
        return None
    logging.info(f"Request to {cc_url} returned successfully. Response code: {resp_code}")

    cc_dict = response.json()
    return cc_dict


"""
Search functions
"""
def find_ticker(df, query_ticker) -> pd.DataFrame:
    return df[df["ticker"] == query_ticker]

def find_title_substring(df, query_title_substring) -> pd.DataFrame:
    return df[df["title"].str.contains(query_title_substring, case=False, na=False)]

def filter_by_column(df, key, query_substrs) -> pd.DataFrame:
    """
    df: company metadata pandas dataframe
    key: name of column we want to filter by (e.g. primaryDocDescription)
    query_substr: accepted values for the column (e.g. ["424b5", "s-3"])
    return: copy of a filtered dataframe
    """
    cum_mask = pd.Series(False, index=df[key].index) # Fill with False values
    for substr in query_substrs:
        curr_mask = df[key].str.contains(substr, case=False, na=False)
        cum_mask = cum_mask | curr_mask
    return df[cum_mask]


def find_dict_key_substr(dict, query_substr) -> List[str]:
    res = []
    for key in dict.keys():
        if query_substr.lower() in key.lower():
            res.append(key)
    return res


