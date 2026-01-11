storage_tracker.py
Gas storage coverage tracker.
Generates an on-demand report on the current state of Czech gas storage using publicly available AGSI data.
Calculations are based on the most recent 14 days and are intended to give a conservative operational snapshot, 
including withdrawal feasibility and compliance with the EU seasonal fill policy.

Requires: wkhtmltopdf (must be installed separately and available on PATH)

Installation:
See https://wkhtmltopdf.org/downloads.html for platform-specific installers.