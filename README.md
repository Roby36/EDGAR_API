
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
