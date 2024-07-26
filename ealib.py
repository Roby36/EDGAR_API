
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
import yfinance as yf
import numpy as np
from typing import Callable
from typing import Tuple

""" UTILITY FUNCTIONS """

""" Timing global constants / variables """
last_req = 0

""" API url getters """
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

""" WRAPPERS """
def requests_get_wrp(url, headers, mrps) -> requests.Response:
    global last_req     # allows function to write to variable outside its scope
    interval = 1/mrps
    curr_time = time.time()
    elapsed_since_last = curr_time - last_req 
    # Wait if the interval since last request is too short
    if elapsed_since_last < interval:
        logging.info(f"Exceeded maximum requests per second. Sleeping for {interval - elapsed_since_last} seconds...")
        time.sleep(interval - elapsed_since_last)
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

def yf_info(ticker_str, query_info):
    """
    Wrapper function to handle keys-not-found errors,
    and usable with lambda to .apply() to whole series.
    Returns None if info not found
    """
    return yf.Ticker(ticker_str).info.get(query_info, None)

def exch_rate(from_currency, to_currency, default=0) -> float:
    """
    Wrapper to convert between two currencies, handling failure from yf

    from_currency:  str for currency we want to convert from   
    to_currency:    str for target currency     
    default:        returned value upon failure in retrieving exchange rate from yf
    return:         exchange rate between from_currency to to_currency                
    """ 

    exch_rate = yf_info(
        f'{from_currency}{to_currency}=X',
        "previousClose"
    )
    if exch_rate is None:
        logging.warning(f'No exchange rate found for {from_currency}/{to_currency}; assuming rate of {default}')
        exch_rate = default
    return exch_rate

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

def get_tickers_df(headers) -> pd.DataFrame:
    resp_dict = get_response_dict(tickers_url(), headers, 1)
    tickers_df = pd.DataFrame.from_dict(resp_dict, orient="index")
    # Pad with leading zeroes the cik, in order to make them 10-digit
    tickers_df["cik_str"] = tickers_df["cik_str"].astype(str).str.zfill(10)
    return tickers_df

def get_sic(tickers_df: pd.DataFrame, ticker: str, req_header, mrps) -> str:
    found_ticker = find_ticker(tickers_df, ticker)
    if found_ticker.empty:
        return None  # or some default value or error handling
    return get_response_dict(metadata_url(found_ticker.iloc[0]["cik_str"]), req_header, mrps=mrps).get("sic")

""" Compunded measures/indicators """
def ocf_average_daily_burn_rate(ocf_df, start_col = "start", end_col = "end", ocf_val_col = "val"):
    # Start by converting start, end columns to datetime
    ocf_df[start_col] = pd.to_datetime(ocf_df[start_col])
    ocf_df[end_col] = pd.to_datetime(ocf_df[end_col])

    # AVG(oper. cash flow / time period)
    return (ocf_df[ocf_val_col] / (ocf_df[end_col] - ocf_df[start_col]).dt.days).mean()

"""Search functions/ getters """
def find_ticker(df: pd.DataFrame, query_ticker: str) -> pd.DataFrame:
    return df[df["ticker"] == query_ticker]

def find_title_substring(df, query_title_substring) -> pd.DataFrame:
    return df[df["title"].str.contains(query_title_substring, case=False, na=False)]

def find_dict_key_substr(dict, query_substrs) -> List[str]:
    res = []
    for key in dict.keys():
        for qs in query_substrs:
            if qs.lower() in key.lower():
                res.append(key)
    return res

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

def filter_filings(filings_df, filing_date_col, form_col, query_forms, max_days, min_days=0) -> pd.DataFrame:
    """
    Function responsible for filtering filings that are then scraped.

    filings_df: company metadata pandas dataframe, converting "filingDate" feature to datetime objects
    query_forms: accepted document types (e.g. ["424b5", "s-3"])
    max_days: maximum days the document dates back from the current date

    return: copy of a filtered dataframe
    """

    # First sort filings by date
    filings_df[filing_date_col] = pd.to_datetime(filings_df[filing_date_col], format='%Y-%m-%d')
    filings_df = filings_df.sort_values(by=filing_date_col, ascending=False)

    # Then initialize the mask
    cum_mask = pd.Series(False, index=filings_df[form_col].index) # Fill with False values

    # Taking the union of the query document types
    for substr in query_forms:
        curr_mask = filings_df[form_col].str.contains(substr, case=False, na=False)
        cum_mask = cum_mask | curr_mask
    
    # Excluding less recent documents
    current_date    = datetime.now()
    max_date_mask   = filings_df[filing_date_col] >= (current_date - timedelta(days=max_days))
    min_date_mask   = filings_df[filing_date_col] <= (current_date - timedelta(days=min_days))
    date_mask   = max_date_mask & min_date_mask
    cum_mask    = cum_mask & date_mask # Combine the existing mask with the new date mask

    return filings_df[cum_mask]


def download_company_filings(req_header, mrps, comp_dir, select_filings, ticker, write_txt, write_pdf):
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
                ticker.get("cik_str", ""),
                curr_doc["accessionNumber"].replace("-",""),
                curr_doc["primaryDocument"]
            ),
            req_header,
            mrps=mrps
        )
        if response == None:
            logging.info(f'Unsuccessful request for document {curr_doc["accessionNumber"]}. Continuing with next document for {comp_dir}')
            continue

        # Before joining paths with os, clean of all "/" symbols!
        curr_form = curr_doc["form"].replace("/","-")
        doc_name = f'{ticker.get("title","")} {curr_form} {curr_doc["filingDate"]}'.replace("/","-")

        # Once source names clean, proceed with directory construction os-work
        curr_dir_full_path = os.path.join(comp_dir, f'{curr_form}')
        os.makedirs(curr_dir_full_path, exist_ok=True)
        curr_form_full_path = os.path.join(curr_dir_full_path, doc_name)

        if write_txt:
            with open(f"{curr_form_full_path}.txt", 'w', encoding='utf-8') as file:
                file.write(response.text)

        if write_pdf:
            pdfkit.from_string(
                response.text, 
                f"{curr_form_full_path}.pdf", 
                options={'enable-local-file-access': ''}
            )


def comp_facts_df(comp_facts, query_fact_substr, sufficient) -> List[Tuple[str, str, str, pd.DataFrame]]:
    """
    Encapsulates the retrieval of historical data for any given company fact, given raw dictionary retrieved from SEC

    comp_facts: raw company fact dictionary retrieved from SEC: can handle None dictionary
    query_fact_substr: substrings for semantic search of company fact
    sufficient: if True, then one string from query_fact_substr sufficient for key match
                if False, then all strings from query_fact_substr must be contained
        True is a good option when both gaap and ifrs convention names are known

    return: [] if desired company fact not located within dictionary
        o/w List of matching tuples (units, acc_standard, match_fact, comp_fact_df), usually more than one
    
    USAGE EXAMPLES: 
    Either run with
        query_fact_substr = ["cash", "operating", "activities"]
        sufficient = False
    for more broad results, or with 
        query_fact_substr = ["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromUsedInOperatingActivities"]
        sufficient = True
    when possible fact names are already known (generally us-gaap or ifrs)
    """
    # Access iteratively nested dictionaries:
    matches_tuple_list = []
    facts_dict = (comp_facts or {}).get("facts", {})
    for as_key, as_dict in facts_dict.items():
        match_fact_keys = find_dict_key_substr(as_dict, query_fact_substr) if sufficient else find_keys_containing_all_substrs(as_dict, query_fact_substr)
        # Here we get a list of the matching key entries found 
        for match_fact in match_fact_keys:
            units_dict = as_dict.get(match_fact, {}).get("units", {})
            if units_dict:
                units = next(iter(units_dict))
                comp_fact_df = pd.DataFrame(units_dict.get(units, {}))
                matches_tuple_list.append((units, as_key, match_fact, comp_fact_df))
    return matches_tuple_list

def comp_fact_avg_change(matches_tuple_list, min_days, max_days) -> float:
    """
    Returns average percentage of the changes of the company fact across datasets provided,
    between averages (0, min_days), (min_days, max_days) of filings available
    """
    deltas = {}
    for units, as_key, match_fact, fact_df in matches_tuple_list:
        # Average value between (min_days, max_days)
        prev_filt_df = filter_filings(fact_df, filing_date_col="end", form_col="form", query_forms=[""], max_days=max_days, min_days=min_days)
        prev_avg = prev_filt_df["val"].mean()

        # Average value between (0, min_days)
        curr_filt_df = filter_filings(fact_df, filing_date_col="end", form_col="form", query_forms=[""], max_days=min_days, min_days=0)
        curr_avg = curr_filt_df["val"].mean()

        # Change in average values
        avg_change = 100 * (curr_avg - prev_avg) / prev_avg
        deltas[match_fact] = avg_change

    # Average of the changes across datasets
    deltas_mean = np.nanmean(np.array(list(deltas.values())))
    return deltas_mean




""" MAIN FILTERING FUNCTION """
def screen_select_companies(
    # general parameters:
        req_header, mrps, tickers_df, root_dir, 
    # filtering parameters:
        query_forms, max_days, max_market_cap, max_ocf_daily_burn_rate, ocf_max_days, ocf_filing_date_col, out_df_sort_key,
    # download parameters:
        write_txt, write_pdf
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Applies hard-coded filters on companies dataframe.
    Downloads selcted documents for dataset of companies, organizing subdirectories
    
    req_header: identification header required by SEC for get requests
    mrps: maximum requests per second to SEC
    tickers_df: dataframe of tickers of companies to scrape
    root_dir: root directory under which all company subdirectories and filings to be saved

    query_forms: list of form types to be considered
    max_days: maximum days for a filing to be considered recent
    max_ocf_burn_rate: maximum cash burn rate value to consider companies
    ocf_max_days: maximum days back to consider OCF filing records
    filing_date_col: how to classify the date of an OCF filing ("end" or "filed")

    write_txt: True iff we want to save downloaded documents in plain text (debugging)
    write_pdf: True iff we want to save downloaded documents in pdf format

    return: dataframe containing companies matching queried parameters, with relevant information
            also saves dataframe to Excel, with CIQ Tickers to gather further information if required
    """

    # Main initializations:
    start_time = time.time()
    os.makedirs(root_dir, exist_ok=True)
    comp_out_df = pd.DataFrame()
    missing_data_df = pd.DataFrame()
    exclusion_reasons = Counter()   # Logging exclusion reasons

    # Iterating through input tickers set
    for index, curr_ticker in tickers_df.iterrows():

        # Initializing company local variables
        curr_comp_series = pd.Series()
        missing_data = False
        comp_name = curr_ticker["title"]
        curr_comp_series["Company name"] = comp_name
        logging.info(f"Starting screening procedure for {comp_name}, at index {index}")
        
        """ (1) FILING FILTER (SEC reqs) """
        curr_comp_mtd = get_response_dict(metadata_url(curr_ticker["cik_str"] ), req_header, mrps=mrps)
        if curr_comp_mtd is None or not curr_comp_mtd.get("filings") or not curr_comp_mtd.get("filings", {}).get("recent"):
            logging.warning(f'Could not find comp_mtd["filings"]["recent"] dictionary for {curr_ticker["ticker"]}, skipping company')
            exclusion_reasons["company filing metadata not found"] += 1
            continue
        curr_filings_df = pd.DataFrame.from_dict(curr_comp_mtd["filings"]["recent"])
        curr_select_filings = filter_filings(curr_filings_df, "filingDate", "form", query_forms, max_days)

        # Filtering condition (1)
        if curr_select_filings.empty:
            logging.info(f"No filings for {comp_name} match the specified criteria. Iterating to next company.")
            exclusion_reasons["no query filing forms found"] += 1
            continue
        # Note: companies with missing filings DISCARDED
        """
        Skipping recent form properties for simplicity:

        for index, val in curr_doc_counts.items():
            comp_out_series[f"Number of recent {index} forms"] = val
        # Dataframe containing most recent forms for each form type 
        # Note: assumes dataframe already sorted by date
        last_forms_df = curr_select_filings.groupby('form').head(1)
        for index, row in last_forms_df.iterrows():
            # Access data from 'accessionNumber' and 'primaryDocument'
            form_name = row["form"]
            comp_out_series[f"Most recent {form_name} date"] = row["filingDate"]
            comp_out_series[f"Most recent {form_name} url"] = doc_url(
                    curr_ticker["cik_str"],
                    row["accessionNumber"].replace("-",""),
                    row["primaryDocument"]
            )
        """

        """ (2) OCF burn FILTER (SEC reqs) """
        # Attempt to retrieve comp_facts dictionary from SEC
        comp_facts = get_response_dict(companyfacts_url(curr_ticker["cik_str"]), req_header, mrps)
        if comp_facts is None:
            logging.warning(f'Failed request when attempting to retrieve company facts for ticker {curr_ticker["ticker"]}, or comp_facts dictionary empty')
        ocf_res = comp_facts_df(
            comp_facts=comp_facts,
            query_fact_substr=["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromUsedInOperatingActivities"], 
            sufficient=True
        )
        if not ocf_res:
            missing_data = True
            logging.info(f"Could not retrieve OCF data for {comp_name}")
        else:
            # NOTE: we select tuple with shortest match_fact (generally the intended company fact)
            ocf_units, res_as_key, ocf_selfact, ocf_df = min(ocf_res, key=lambda x: len(x[2]))
            curr_comp_series["OCF Currency"]        = ocf_units
            curr_comp_series["OCF Name"]            = ocf_selfact # mostly for debugging
            curr_comp_series["Accounting Standard"] = res_as_key
            ocf_df_filt = filter_filings(ocf_df, 
                filing_date_col=ocf_filing_date_col, form_col="form", query_forms=[""], max_days=ocf_max_days
            )
            if not ocf_df_filt.empty:
                curr_comp_series["Avg daily OCF burn"] = ocf_average_daily_burn_rate(ocf_df_filt)
                # Exhange work
                curr_comp_series["USD Avg daily OCF burn"] = curr_comp_series["Avg daily OCF burn"] * exch_rate(curr_comp_series["OCF Currency"], "USD")

                # Filtering condition (2)
                if curr_comp_series["USD Avg daily OCF burn"] > max_ocf_daily_burn_rate:
                    logging.info(f'Excluding {comp_name} with USD Avg daily OCF burn {curr_comp_series["USD Avg daily OCF burn"]}')
                    exclusion_reasons["OCF burn rate"] += 1
                    continue
            else:
                logging.warning(f'No OCF records found for {comp_name} within {ocf_max_days} days')
                missing_data = True
        # Note: companies WITHOUT OCF information SPARED

        """ (3) MARKET CAP FILTER (yf reqs) """
        curr_comp_series["Market Cap"]          = yf_info(curr_ticker["ticker"], "marketCap")
        curr_comp_series["Market Cap Currency"] = yf_info(curr_ticker["ticker"], "currency")
        if curr_comp_series["Market Cap Currency"] is None or curr_comp_series["Market Cap"] is None:
            missing_data = True
        # Ensure market cap converted to USD for fair comparison
        else:
            # Exchange work
            curr_comp_series["USD Market Cap"] = curr_comp_series["Market Cap"] * exch_rate(curr_comp_series["Market Cap Currency"], "USD")

            # Filtering condition (3)
            if curr_comp_series["USD Market Cap"] > max_market_cap:
                logging.info(f'Excluding company {comp_name} with USD market cap {curr_comp_series["USD Market Cap"]}')
                exclusion_reasons["Market cap"] += 1
                continue
        # Note: companies WITH missing market cap data SPARED


        """ Other deriving METRICS / Properties """
        curr_comp_series["CIQ ticker"] = ciq_ticker(curr_comp_mtd, curr_ticker)
        if "USD Market Cap" in curr_comp_series and curr_comp_series["USD Market Cap"] is not None and "USD Avg daily OCF burn" in curr_comp_series:
                    curr_comp_series["Avg yearly OCF burn / Market Cap"] = 360 * curr_comp_series["USD Avg daily OCF burn"] / curr_comp_series["USD Market Cap"]

        """ Adding company series to cumulative dataframes, based on whether all information was received """
        if missing_data:
            missing_data_df = pd.concat([missing_data_df, pd.DataFrame([curr_comp_series])], ignore_index=False)
        else:
            comp_out_df     = pd.concat([comp_out_df,     pd.DataFrame([curr_comp_series])], ignore_index=False)

        """ FINAL: Saving company selected fiings after ALL filtering has been done """
        comp_dir = os.path.join(root_dir, comp_name)
        download_company_filings(req_header, mrps, comp_dir, curr_select_filings, curr_ticker["cik_str"] , write_txt=write_txt, write_pdf=write_pdf)
        logging.info(f"Downloaded filings for {comp_name} at index {index}")
    

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
    logging.info(f'Exclusion reasons: {exclusion_reasons}')

    # Sort output dataframe
    if out_df_sort_key in comp_out_df.columns:
        comp_out_df = comp_out_df.sort_values(by=out_df_sort_key, ascending=True)
    else:
        logging.warning(f"'{out_df_sort_key}' not found in DataFrame. Sorting not performed.")
    
    return (comp_out_df, missing_data_df)





""" Table cleaning functions """

def subcolumns(df: pd.DataFrame, col: pd.Series) -> List:
    """
    Returns a mask, where False means that a column in the dataframe is 
    a "subcolumn" of col, meaning that it contains the same data but more NaN values
    """
    mask = []
    for index, dcol in df.items():
        is_equal = col.isna() | dcol.isna() | (col == dcol)
        mask.append(~((is_equal == 1).all()) or col.isna().sum() >= dcol.isna().sum())
    return mask

def discard_subcolumns(df : pd.DataFrame) -> pd.DataFrame:
    """ 
    Applies the subcolumns function on each column of the dataframe and returns the resulting dataframe
    """
    cum_mask = [True for _ in range(len(df.columns))] # initialize uniformly to True
    for index, col in df.items():
        mask = subcolumns(df, col)
        cum_mask = [cum_mask[i] and mask[i] for i in range(len(cum_mask))] # vectorized AND operation
    return df.loc[:, cum_mask]

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ (1) Remove fully NaN rows and columns """
    res = df.dropna(how='all').dropna(axis=1, how='all')

    """ (2) Remove duplicate columns """
    df_transposed = res.T
    df_transposed = df_transposed[~df_transposed.duplicated()]
    res = df_transposed.T

    """ (3) TODO: Convert Excel-style numbering to interpretable numbers """
    res = res.map(lambda x: x.replace('—', '0') if isinstance(x, str) else x)  # Replace dashes with zero (assumes dashes mean zero)
    res = res.map(lambda x: pd.to_numeric(x.replace(',', '').replace('(', '-').replace(')', '') if isinstance(x, str) else x, errors="ignore")) # TODO: Handle errors explicitly (warning)

    """ (4) IMPORTANT: Discard NaN subcolumns AT THE END! """
    res = discard_subcolumns(res)

    """ TODO: other preprocessing steps """
    return res


def clean_tabs(
        dfs: List[pd.DataFrame], 
        clean_func: Callable[[pd.DataFrame], pd.DataFrame], 
        remove_small_tabs: bool = True
    ) -> List[pd.DataFrame]:
    """ 
    Overall cleaning function, taking a list of datatables 
    and retuning the fully refined list
    """
    res_dfs = []
    for df in dfs:
        """ First clean this df """
        res_df = clean_func(df)

        """ Then start applying various filters """
        if remove_small_tabs and res_df.shape[0] <= 1 and res_df.shape[1] <= 2:
            continue
        
        """ Finally save the dataframe to the list"""
        res_dfs.append(res_df)
    
    return res_dfs


def df_contains_substr(df: pd.DataFrame, query_str: str) -> bool:
    return  df.map(lambda x: query_str.lower() in str(x).lower()).any().any()

def df_contains_substr_all(df: pd.DataFrame, queries: List[str]) -> bool:
    return all(df_contains_substr(df, query_str) for query_str in queries)
