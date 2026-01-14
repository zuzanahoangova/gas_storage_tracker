# EU Gas Storage Compliance Tracker

Python tool for evaluating the current state of Czech gas storage under the amended EU Gas Storage Regulation (EU/2022/1032, post-2025 regime).

The script translates regulatory storage targets into operational constraints and assesses whether the system remains on track under recent injection and withdrawal trends.

## What the tool does

Using publicly available AGSI data, the tracker provides a conservative, decision-oriented snapshot of the gas storage system by:

- evaluating compliance with the seasonal 90 % storage fill target within the regulatory window (1 October–1 December),
- estimating remaining days of system coverage under current withdrawal rates,
- assessing feasibility and risk margins using 7-day and 14-day rolling averages of injection and withdrawal flows.

## Outputs

- PDF summary report with key indicators and policy verdicts  
- Excel file containing input data and calculated parameters

Reports distinguish between seasonal regimes (summer, winter) and adapt calculations accordingly.

## Data source

- Aggregated Gas Storage Inventory (AGSI), operated by Gas Infrastructure Europe (GIE)  
  https://agsi.gie.eu

## Requirements

- Python 3.x
- wkhtmltopdf (must be installed separately and available on PATH: https://wkhtmltopdf.org/downloads.html)

## Notes

This tool is intended for analytical and exploratory use.  
It does not represent an official forecast or operational instruction.
