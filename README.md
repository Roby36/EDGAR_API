
# Simple EDGAR Scraper

## Introduction
Lightweight Python-based tool designed to efficiently scrape and download filings from the EDGAR database, which is provided by the U.S. Securities and Exchange Commission (SEC). The tool leverages the EDGAR API to access real-time filing data, utilizing various endpoints to fetch metadata, company facts, and specific documents based on user-defined criteria.

## Main end points used:
- **`https://www.sec.gov/files/company_tickers.json`**: Provides the list in .json format of all companies currently listed on SEC.
- **`https://data.sec.gov/submissions/CIK{cik}.json`**: Provides the metadata for the company with CIK number given by `{cik}`, including recent filing accession numbers and other information.
- **`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`**: Provides company facts dataset for the company with CIK number given by `{cik}` as presented on all financial statements ever published on SEC by the company.
- **`https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}{concept}.json`**: Provides more insights on a particular company concept or financial metric.
- **`https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_stripped}/{filename}`**: Provides direct access to the htm-formatted version of an archived filing with dash-free accession number `{accession_number_stripped}` and document name `{filename}` for company with CIK number `{cik}`.

## Main Features
- **Request Throttling**: Implements controlled request timings to comply with the SEC's rate limits, ensuring the scraper runs efficiently without violating usage terms. Response codes are handled and logged to the user. 
- **Company Filings Download**: Automates the process of downloading specified types of SEC filings for multiple companies within defined time constraints.
- **Document Handling**: Saves the htm filings in both text and PDF formats to a specifiable root_directory, making it suitable for both immediate review and archival purposes.
- **Dynamic Searching**: Allows users to filter and download documents based on ticker symbols, specific form types (like 10-K, 10-Q), and date ranges.

## Main input parameters
The main function of the module, `screen_select_companies`, performs a screening on a subset of the SEC database, currently based on the parameters and criteria below:
### General/EDGAR parameters
- **req_header**: header to identify with SEC EDGAR database.
- **mrps**: maximum requests per second. This should be `< 10` to comply with current restrictions.
- **tickers_df**: `pd.DataFrame` object holding the subset of tickers we want to perform our screening on. The entire set can be retrieved from SEC through the function call `get_tickers_df(req_header)`, and then sliced accordingly. 
- **root_dir**: Path where to store the root folder of the downloaded filings tree. The function generates subdirectories named after each company that files are downloaded for. 
### Currently supported filtering parameters
- **query_forms**: The list of form types that a company is required to have filed to SEC to be considered for the screening, on a sufficient basis (i.e. at least one of the forms is sufficient for the company to be screened).
- **max_days**: The maximum number of days from the current date within which a filing is taken under consideration. If none of the filings fall within the desired range, then the company is discarded. 
- **max_market_cap**: The threshold market cap beyond which companies are discarded, in USD. Retrieved market caps are converted from their local currencies to USD before the comparison.
- **max_ocf_daily_burn_rate**: Maximum daily operating cash flow burn rate for a company to be considered, in USD. Note that positive burn rate is interpreted as positive cash flow, hence companies beyond this threshold will not be considered. For example, if set to zero, only companies with negative daily burn rates (i.e. negative operating cash flows) will be considered. Retrieved data from SEC is converted to USD before performing the comparison. 
- **ocf_max_days**: Maximum days from current date within which operating cash flow statements are considered in the computation of an average cash burn rate for the period.  
- **ocf_filing_date_col**: If set to `"filed"`, then the function applies `ocf_max_days` on the filing date of the OCF filings. This yields more data to calculate the average rate on, with the drawback that some filings may effectively cover a more dated period than expected. On the other hand, if set to `"end"`, then the function applies `ocf_max_days` on the effective accounting end date of the OCF filings. This guarantees more recent results, but usually reduces to only one or very few filings. 
### Output parameters
- **out_df_sort_key**: The key by which we want to rank the output results. For example, we may enter `"Avg yearly OCF 
burn / Market Cap"`, which is computed by the function
- **write_txt**: Set to `True` if we want to save the filings as text files (mainly for debugging).
- **write_pdf**: Set to `True` if we want to save the filings in a rendered PDF format (generally preferred option). 

