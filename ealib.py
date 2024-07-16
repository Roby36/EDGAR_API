
import requests
import logging
import pandas as pd
from typing import List
import time 
from datetime import datetime, timedelta
import os
import pdfkit

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
Search functions
"""
def find_ticker(df, query_ticker) -> pd.DataFrame:
    return df[df["ticker"] == query_ticker]

def find_title_substring(df, query_title_substring) -> pd.DataFrame:
    return df[df["title"].str.contains(query_title_substring, case=False, na=False)]

def filter_filings(filings_df, query_forms, max_days) -> pd.DataFrame:
    """
    Function responsible for filtering filings that are then scraped.

    filings_df: company metadata pandas dataframe, assuming "filingDate" feature already converted to datetime objects
    query_forms: accepted document types (e.g. ["424b5", "s-3"])
    max_days: maximum days the document dates back from the current date

    return: copy of a filtered dataframe
    """

    # Taking the union of the query document types
    cum_mask = pd.Series(False, index=filings_df["form"].index) # Fill with False values
    for substr in query_forms:
        curr_mask = filings_df["form"].str.contains(substr, case=False, na=False)
        cum_mask = cum_mask | curr_mask
    
    # Excluding less recent documents
    current_date = datetime.now()
    date_mask = filings_df['filingDate'] >= (current_date - timedelta(days=max_days))
    cum_mask = cum_mask & date_mask # Combine the existing mask with the new date mask

    return filings_df[cum_mask]

def find_dict_key_substr(dict, query_substr) -> List[str]:
    res = []
    for key in dict.keys():
        if query_substr.lower() in key.lower():
            res.append(key)
    return res

def ciq_ticker(comp_mtd) -> str:
    exch = comp_mtd["exchanges"][0]
    tck  = comp_mtd["tickers"][0]
    ciqt = f"{exch}:{tck}"
    return ciqt

"""
Main document scraping functions 
"""

def get_company_entry(ticker, comp_mtd, select_filings, cik) -> pd.Series:
    """ 
    Wrapper processing company through metadata and filtered filings to make an entry for the company
    """

    comp_out_series = pd.Series()
    comp_out_series["title"]  = ticker["title"]
    comp_out_series["CIQ ticker"] = ciq_ticker(comp_mtd)
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
                cik,
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

        # Initialize current ticker details 
        curr_cik = curr_ticker["cik_str"] 
        curr_comp_mtd = get_response_dict(metadata_url(curr_cik), req_header, mrps=mrps)
        curr_filings_df = pd.DataFrame.from_dict(curr_comp_mtd["filings"]["recent"])

        # Sort documents by filing date
        curr_filings_df['filingDate'] = pd.to_datetime(curr_filings_df['filingDate'], format='%Y-%m-%d')
        curr_filings_df = curr_filings_df.sort_values(by='filingDate', ascending=False)

        # Apply filing filters, and continue if they yield an empty dataframe
        curr_select_filings = filter_filings(curr_filings_df, query_forms, max_days)
        if curr_select_filings.empty:
            logging.info(f"No filings for {comp_name} match the specified criteria. Iterating to next company.")
            continue

        # Add company data as row to overall output dataframe
        curr_comp_out_series = get_company_entry(curr_ticker, curr_comp_mtd, curr_select_filings, curr_cik)
        comp_out_df = pd.concat([comp_out_df, pd.DataFrame([curr_comp_out_series])], ignore_index=False)

        # Scrape files for the company
        comp_dir = os.path.join(root_dir, comp_name)
        download_company_filings(req_header, mrps, comp_dir, curr_select_filings, curr_cik, write_txt=write_txt, write_pdf=write_pdf)
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

