
import requests
import logging
import pandas as pd
from typing import List
from typing import Tuple
import time 
from datetime import datetime, timedelta
import os
import pdfkit
from collections import Counter
import random

"""
Timing global constants / variables:
"""
last_req = 0

""" 
UTILS
"""
def requests_get_wrp(url, headers, mrps) -> requests.Response:
    global last_req     # allows function to write to variable outside its scope
    interval = 1/mrps
    curr_time = time.time()
    elapsed_since_last = curr_time - last_req 
    # Wait if the interval since last request is too short
    if elapsed_since_last < interval:
        time.sleep(interval - elapsed_since_last)
        logging.info(f"Exceeded maximum requests per second. Sleeping for {interval - elapsed_since_last} seconds...")
    response = requests.get(url, headers=headers)
    # Update the last request timestamp
    last_req = time.time()

    # Handle response
    resp_code = response.status_code 
    if resp_code < 200 or resp_code >= 300:
        logging.warning(f"Request to {url} was unsuccessful. Response code: {resp_code}")
        return None
    logging.info(f"Request to {url} returned successfully. Response code: {resp_code}")
    return response

def get_response_dict(url, headers, mrps) -> dict:
    response = requests_get_wrp(url, headers, mrps)
    if response == None:
        return None
    return response.json()

""" 
API url getters
"""
def tickers_url() -> str:
    return "https://www.sec.gov/files/company_tickers.json"

def metadata_url(cik) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"

def companyfacts_url(cik) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

def companyconcept_url(cik, concept) -> str:
    """
    COMPANY-CONCEPT url getter
        e.g. concept = /us-gaap/Revenues
    """
    return f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}'f'{concept}.json'

def doc_url(cik, accession_number_stripped, filename) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_stripped}/{filename}"

"""
COMPANY TICKERS dataframe getter
"""
def get_tickers_df(headers) -> pd.DataFrame:
    resp_dict = get_response_dict(tickers_url(), headers, 1)
    tickers_df = pd.DataFrame.from_dict(resp_dict, orient="index")
    # Pad with leading zeroes the cik, in order to make them 10-digit
    tickers_df["cik_str"] = tickers_df["cik_str"].astype(str).str.zfill(10)
    return tickers_df

""" 
Compunded measures 
"""
def ocf_average_daily_burn_rate(ocf_df, start_col = "start", end_col = "end", ocf_val_col = "val"):
    # Start by converting start, end columns to datetime
    ocf_df[start_col] = pd.to_datetime(ocf_df[start_col])
    ocf_df[end_col] = pd.to_datetime(ocf_df[end_col])

    # AVG(oper. cash flow / time period)
    return (ocf_df[ocf_val_col] / (ocf_df[end_col] - ocf_df[start_col]).dt.days).mean()


"""
Search functions
"""
def find_ticker(df, query_ticker) -> pd.DataFrame:
    return df[df["ticker"] == query_ticker]

def find_title_substring(df, query_title_substring) -> pd.DataFrame:
    return df[df["title"].str.contains(query_title_substring, case=False, na=False)]

def filter_filings(filings_df, filing_date_col, form_col, query_forms, max_days) -> pd.DataFrame:
    """
    Function responsible for filtering filings that are then scraped.

    filings_df: company metadata pandas dataframe, converting "filingDate" feature to datetime objects
    query_forms: accepted document types (e.g. ["424b5", "s-3"])
    max_days: maximum days the document dates back from the current date

    return: copy of a filtered dataframe
    """

    # Taking the union of the query document types
    cum_mask = pd.Series(False, index=filings_df[form_col].index) # Fill with False values
    for substr in query_forms:
        curr_mask = filings_df[form_col].str.contains(substr, case=False, na=False)
        cum_mask = cum_mask | curr_mask
    
    # Sort by date
    filings_df[filing_date_col] = pd.to_datetime(filings_df[filing_date_col], format='%Y-%m-%d')
    filings_df = filings_df.sort_values(by=filing_date_col, ascending=False)
    
    # Excluding less recent documents
    current_date = datetime.now()
    date_mask = filings_df[filing_date_col] >= (current_date - timedelta(days=max_days))
    cum_mask = cum_mask & date_mask # Combine the existing mask with the new date mask

    return filings_df[cum_mask]

def find_dict_key_substr(dict, query_substrs) -> List[str]:
    res = []
    for key in dict.keys():
        for qs in query_substrs:
            if qs.lower() in key.lower():
                res.append(key)
    return res

# More sophisticated dictionary key search 
def find_keys_containing_all_substrs(dict, query_substrs) -> str: 
    """
    Extension of function above.
    Intended to identify metric by checking for all substrings containment
    """
    res = []
    for key in dict.keys():
        not_found = False
        for qs in query_substrs:
            if qs.lower() not in key.lower():
                not_found = True
                break
        if not not_found:
            res.append(key)
    return res

def ciq_ticker(comp_mtd, ticker) -> str:
    """ 
    Attempts to retrieve {exchange}:{ticker} from company metadata
    Returns {ticker} as failsafe
    """
    exch = comp_mtd["exchanges"]
    tck  = comp_mtd["tickers"]
    if not exch or not tck:
        return ticker["ticker"]
    return f"{exch[0]}:{tck[0]}"



"""
Main document scraping functions 
"""

def get_company_entry(ticker, comp_mtd, select_filings) -> pd.Series:
    """ 
    Wrapper processing information to store for each company
    """

    comp_out_series = pd.Series()
    comp_out_series["title"]  = ticker["title"]
    comp_out_series["CIQ ticker"] = ciq_ticker(comp_mtd, ticker)
    curr_doc_counts = select_filings["form"].value_counts()
    for index, val in curr_doc_counts.items():
        comp_out_series[f"Number of recent {index} forms"] = val

    # Dataframe containing most recent forms for each form type 
    # Note: assumes dataframe already sorted by date
    last_forms_df = select_filings.groupby('form').head(1)

    for index, row in last_forms_df.iterrows():
        # Access data from 'accessionNumber' and 'primaryDocument'
        form_name = row["form"]
        comp_out_series[f"Most recent {form_name} date"] = row["filingDate"]
        comp_out_series[f"Most recent {form_name} url"] = doc_url(
                ticker["cik_str"],
                row["accessionNumber"].replace("-",""),
                row["primaryDocument"]
        )

    return comp_out_series


def download_company_filings(req_header, mrps, comp_dir, select_filings, cik, write_txt, write_pdf):
    """ 
    Downloads selcted documents for a company, organizing subdirectories.
    
    req_header: identification header required for get requests
    mrps: maximum requests per second to SEC
    comp_dir: root directory where company filings to be saved
    select_filings: filtered dataframe of filings to download
    cik: string with company's cik identifier
    write_txt: True iff we want to save downloaded documents in plain text (debugging)
    write_pdf: True iff we want to save downloaded documents in pdf format
    """

    os.makedirs(comp_dir, exist_ok=True)
    for index, curr_doc in select_filings.iterrows():
        response = requests_get_wrp(
            doc_url(
                cik,
                curr_doc["accessionNumber"].replace("-",""),
                curr_doc["primaryDocument"]
            ),
            req_header,
            mrps=mrps
        )
        if response == None:
            logging.info(f'Unsuccessful request for document {curr_doc["accessionNumber"]}. Continuing with next document for {comp_dir}')
            continue

        curr_dir_full_path = os.path.join(comp_dir, curr_doc["form"])
        os.makedirs(curr_dir_full_path, exist_ok=True)
        curr_form_full_path = os.path.join(curr_dir_full_path, curr_doc["accessionNumber"])

        if write_txt:
            with open(f"{curr_form_full_path}.txt", 'w', encoding='utf-8') as file:
                file.write(response.text)

        if write_pdf:
            pdfkit.from_string(
                response.text, 
                f"{curr_form_full_path}.pdf", 
                options={
                    'no-images': '',  
                    'disable-external-links': '',
                    'disable-internal-links': ''  
                }
            )

def download_select_filings(req_header, mrps, tickers_df, root_dir, query_forms, max_days, write_txt, write_pdf) -> pd.DataFrame:
    """ 
    Downloads selcted documents for dataset of companies, organizing subdirectories
    
    req_header: identification header required by SEC for get requests
    mrps: maximum requests per second to SEC
    tickers_df: dataframe of tickers of companies to scrape
    root_dir: root directory under which all company subdirectories and filings to be saved
    query_forms: list of form types to be considered
    max_days: maximum days for a filing to be considered recent
    write_txt: True iff we want to save downloaded documents in plain text (debugging)
    write_pdf: True iff we want to save downloaded documents in pdf format

    return: dataframe containing companies matching queried parameters, with relevant information
            also saves dataframe to Excel, with CIQ Tickers to gather further information if required
    """

    start_time = time.time()
    os.makedirs(root_dir, exist_ok=True)
    comp_out_df = pd.DataFrame()

    for index, curr_ticker in tickers_df.iterrows():

        comp_name = curr_ticker["title"]
        logging.info(f"Starting download filings procedure for {comp_name}")

        """ 
        Requesting company metadata to filter on filings
        """
        curr_comp_mtd = get_response_dict(metadata_url(curr_ticker["cik_str"] ), req_header, mrps=mrps)
        curr_filings_df = pd.DataFrame.from_dict(curr_comp_mtd["filings"]["recent"])

        # Apply filing filters, and continue if they yield an empty dataframe
        curr_select_filings = filter_filings(curr_filings_df, "filingDate", "form", query_forms, max_days)
        if curr_select_filings.empty:
            logging.info(f"No filings for {comp_name} match the specified criteria. Iterating to next company.")
            continue

        """ 
        TODO: other filtering before saving and downloading company
        """

        """ 
        Saving company properties to overall dataframe
        """
        curr_comp_out_series = get_company_entry(curr_ticker, curr_comp_mtd, curr_select_filings)
        comp_out_df = pd.concat([comp_out_df, pd.DataFrame([curr_comp_out_series])], ignore_index=False)

        """ 
        Downloading select filings for the company --> data/computationally intensive
        """
        comp_dir = os.path.join(root_dir, comp_name)
        download_company_filings(req_header, mrps, comp_dir, curr_select_filings, curr_ticker["cik_str"] , write_txt=write_txt, write_pdf=write_pdf)
        logging.info(f"Terminating download filings procedure for {comp_name}")
    

    # End of main loop. Timing work:
    end_time = time.time()
    duration_seconds = end_time - start_time
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    seconds = duration_seconds % 60
    logging.info(
        f"Screened a total of {tickers_df.shape[0]} companies in " 
        f"{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"
    )

    # Save to Excel and output dataframe 
    comp_out_df.to_excel(f'{root_dir}.xlsx', sheet_name='Sheet1', index=False)
    return comp_out_df


def company_fact_df(curr_ticker, as_keys, query_fact_substr, sufficient, req_header, mrps) -> (str, str, pd.DataFrame):
    """
    Encapsulates the request and retrieval of historical data for any given company fact.

    curr_ticker: fixed ticker (fucntion expected to be called within iterative loop)
    company_facts_keys: the keys to be searched within company facts to find desired fact
    query_fact_substr: substrings for semantic search of company fact
    sufficient: if True, then one string from query_fact_substr sufficient for key match
                if False, then all strings from query_fact_substr must be contained
        True is a good option when both gaap and ifrs convention names are known

    req_header, mrps: request settings passed on

    return: None if the company facts request fails, or company fact not found, 
        o/w (units, comp_fact, pd DataFrame) containing historical data for requested company fact
    
    USAGE EXAMPLES: 
    Either run with
        query_fact_substr = ["cash", "operating", "activities"]
        sufficient = False
    for more broad results, or with 
        query_fact_substr = ["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromUsedInOperatingActivities"]
        sufficient = True
    when possible fact names are already known (generally us-gaap or ifrs)
    """

    # Attempt to retrieve comp_facts dictionary from SEC
    comp_facts = get_response_dict(companyfacts_url(curr_ticker["cik_str"]), req_header, mrps)
    if comp_facts is None or not comp_facts or "facts" not in comp_facts:
        logging.warning(f'Failed request when attempting to retrieve company facts for ticker {curr_ticker["ticker"]}, or comp_facts dictionary empty')
        return None
    
    # Attempt to find the appropriate accounting standards key for the desired company fact 
    as_key_found = False
    cf_key = []
    res_as_key = ""
    for key in as_keys:
        if key in comp_facts["facts"]:
            as_key_found = True
            res_as_key = key
            # Choose comparator to search for company fact key
            if sufficient:
                cf_key = find_dict_key_substr(comp_facts["facts"][res_as_key], query_fact_substr)
            else: 
                cf_key = find_keys_containing_all_substrs(comp_facts["facts"][res_as_key], query_fact_substr)
            break
    if not as_key_found or not cf_key:
        logging.warning(f'Accounting standards key not found for ticker {curr_ticker["ticker"]}, or query fact not found')
        return None
    if len(cf_key) > 1:
        logging.warning(f'More than one company fact for {curr_ticker["ticker"]} matches search query; returning shortest') 
    min_cf_key = min(cf_key, key=len)

    # Attempt to access inner dictionaries to find historical values
    nested_cf_dict = comp_facts["facts"][res_as_key][min_cf_key]
    if "units" not in nested_cf_dict or not nested_cf_dict["units"]:
        logging.warning(f'Cannot find nested dictionary company facts for ticker {curr_ticker["ticker"]}, comp fact key {min_cf_key}')
        return None
    units = next(iter(nested_cf_dict["units"])) # Note: we just retrieve some unit, not checking whether more are available
    return (units, min_cf_key, pd.DataFrame(nested_cf_dict["units"][units]))




