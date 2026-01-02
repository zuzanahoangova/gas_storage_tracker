from datetime import date, timedelta
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from enum import Enum


class SeasonMode(Enum):
    TRANSITION = 0  # transition oct 2 - dec 1
    WINTER = 1      # winter mode dec 2 - march 30
    SUMMER = 2      # summer mode march 31 - oct 1


# pick op season_mode
def decide_season_mode(date_today):
    m, d = date_today.month, date_today.day
    if (m == 3 and d == 31) or (4 <= m <= 9) or (m == 10 and d == 1):
        return SeasonMode.SUMMER
    elif (m == 12 and d >= 2) or (m in [1, 2]) or (m == 3 and d <= 30):
        return SeasonMode.WINTER
    else:
        return SeasonMode.TRANSITION


# load agsi api key
def load_api_key():
    load_dotenv()  # reads .env into memory
    api_key = os.getenv("AGSI_API_KEY")     # gets environment variables
    # load_dotenv() looks for .env in the same path as this file, gotta specify path if testing in console

    if not api_key:
        raise RuntimeError("AGSI API key not found.")
    return api_key


# fetch data from agsi
def fetch_data(date_today, season_mode) -> list[dict]:
    base_url = "https://agsi.gie.eu/api"

    # search params
    end_date = date_today - timedelta(days=1)
    if season_mode is SeasonMode.TRANSITION:   # fetches 2 months of data for transition period, otherwise 2 weeks
        start_date = date_today - timedelta(days=62)
    else:
        start_date = date_today - timedelta(days=15)

    params = {
        "country": "CZ",
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
    }

    # authentication
    api_key = load_api_key()
    headers = {"x-key": api_key}

    # data fetch
    r = requests.get(base_url, params=params, headers=headers)
    r.raise_for_status()

    payload = r.json()
    raw_json = payload["data"] if isinstance(payload, dict) else payload
    return raw_json


# convert json data to pandas dataframe
def to_dataframe(raw_json: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(raw_json)
    return data


# check for issues in data
def check_data(data, date_today):
    issues = []
    dates = pd.to_datetime(data['gasDayEnd'])
    end_date = pd.Timestamp(date_today - timedelta(days=1))

    # checks latest date
    if dates.iloc[0] != end_date:
        issues.append(f"Latest gasDayEnd is {dates.iloc[0].date()}, expected {end_date.date()}.")

    # checks missing days
    expected = pd.date_range(start=dates.min(), end=dates.max(), freq="D")
    missing = expected.difference(dates)
    if not missing.empty:
        issues.append(f"Missing {len(missing)} gas days: {missing.date.tolist()}.")
    return issues


# clean data
def extract_data(data):
    st_spec = pd.DataFrame({
        "gas_day": [data.loc[0, 'gasDayEnd']],               # current day as string
        "tech_capacity": [data.loc[0, 'workingGasVolume']],  # in TWh
        "current_storage": [data.loc[0, 'gasInStorage']],    # in TWh
    })
    st_flow = data.loc[:14, ['injection', 'withdrawal']]       # in GWh
    return st_spec, st_flow


def calculate_summer(data):
    st_spec, st_flow = extract_data(data)
    current_storage = st_spec.loc[0, "current_storage"]     # in TWh
    tech_capacity = st_spec.loc[0, "tech_capacity"]         # in TWh
    gas_day = pd.to_datetime(st_spec.loc[0, "gas_day"])     # current gas day

    # policy: hit 90% once between oct 1 and dec 1
    factor = 0.9  # policy goal
    nov1 = pd.Timestamp(gas_day.year, 11, 1)  # nov 1: anchor date, old policy 2022-2025
    dec1 = pd.Timestamp(gas_day.year, 12, 1)  # dec 1: hard deadline, new policy after april 2025
    days_to_nov1 = (nov1 - gas_day).days
    days_to_dec1 = (dec1 - gas_day).days

    injection7 = st_flow["injection"].iloc[:7].mean()
    injection14 = st_flow["injection"].iloc[:14].mean()
    withdrawal7 = st_flow["withdrawal"].iloc[:7].mean()
    withdrawal14 = st_flow["withdrawal"].iloc[:14].mean()

    # all following flows in GWh
    # calculations assume net injection regime (injection > withdrawal)

    # How many days left until policy goal fulfilled?
    days_to_goal7 = (tech_capacity * factor - current_storage) * 1000 / (injection7 - withdrawal7)
    days_to_goal14 = (tech_capacity * factor - current_storage) * 1000 / (injection14 - withdrawal14)
    goal_date7 = gas_day + pd.Timedelta(days=np.ceil(days_to_goal7))
    goal_date14 = gas_day + pd.Timedelta(days=np.ceil(days_to_goal14))

    # What minimal injection rate is needed to reach policy goal based on recent withdrawal rate?
    min_inj_to_nov1_7 = (tech_capacity * factor - current_storage) * 1000 / days_to_nov1 + withdrawal7
    min_inj_to_nov1_14 = (tech_capacity * factor - current_storage) * 1000 / days_to_nov1 + withdrawal14
    min_inj_to_dec1_7 = (tech_capacity * factor - current_storage) * 1000 / days_to_dec1 + withdrawal7
    min_inj_to_dec1_14 = (tech_capacity * factor - current_storage) * 1000 / days_to_dec1 + withdrawal14

    summer_data = pd.DataFrame(
        {
            "avg_7d": [injection7, withdrawal7, days_to_goal7, goal_date7,
                       min_inj_to_nov1_7, min_inj_to_dec1_7],
            "avg_14d": [injection14, withdrawal14, days_to_goal14, goal_date14,
                        min_inj_to_nov1_14, min_inj_to_dec1_14],
        },
        index=["injection", "withdrawal", "days", "date", "min_inj_nov1", "min_inj_dec1"]
    )

    policy_dates = pd.DataFrame(
        {
            "nov1": [nov1, days_to_nov1],
            "dec1": [dec1, days_to_dec1],
        },
        index=["date", "span"]
    )
    return summer_data, policy_dates


def calculate_winter(data):
    st_spec, st_flow = extract_data(data)
    current_storage = st_spec.loc[0, "current_storage"] * 1000      # in GWh
    gas_day = pd.to_datetime(st_spec.loc[0, "gas_day"])             # current gas day

    # end-of-winter date (convention: March 31)
    if gas_day.month < 4:
        eow = pd.Timestamp(gas_day.year, 3, 31)
    else:
        eow = pd.Timestamp(gas_day.year + 1, 3, 31)

    days_to_eow = (eow - gas_day).days
    max_withdrawal_eow = current_storage / days_to_eow
    withdrawal7 = st_flow["withdrawal"].iloc[:7].mean()
    withdrawal14 = st_flow["withdrawal"].iloc[:14].mean()

    # How many days of coverage left assuming no more injections?
    days_of_cover7 = current_storage / withdrawal7
    days_of_cover14 = current_storage / withdrawal14

    if np.isinf(days_of_cover7):
        end_date7 = pd.NaT  # no end date if 0 withdrawal over 7 days
    else:
        end_date7 = gas_day + pd.Timedelta(days=np.floor(days_of_cover7))

    if np.isinf(days_of_cover14):
        end_date14 = pd.NaT
    else:
        end_date14 = gas_day + pd.Timedelta(days=np.floor(days_of_cover14))

    # columns = scenarios, rows = concepts
    winter_data = pd.DataFrame(
        {
            "ref": [eow, days_to_eow, max_withdrawal_eow],
            "avg_7d": [end_date7, days_of_cover7, withdrawal7],
            "avg_14d": [end_date14, days_of_cover14, withdrawal14],
        },
        index=["date", "days", "withdrawal"]
    )
    return winter_data


# check if policy goal (storage >=90%) reached on any day in dataset
def check_policy_goal(data):
    goal_reached = (data['full'] >= 90).any()
    return goal_reached   # true/false


# determine the earliest day from dataset when 90% reached
def first_hit_date(data):
    hit_date = data.loc[data['full'] >= 90, 'gasDayEnd'].min()
    return hit_date


# decides what to calculate
def decide_calculation(season_mode, goal_reached):
    if season_mode is SeasonMode.WINTER:
        return 'W'
    elif season_mode is SeasonMode.SUMMER:
        return None if goal_reached else 'S'
    else:
        return 'W' if goal_reached else 'S'


# calculates outputs
def calculate_report_content(report_mode, data):
    """
        Returns:
          - winter: (winter_data)
          - summer active: (summer_data, policy_dates)
          - summer complete: (first_hit_date)
    """
    if report_mode == 'W':
        return calculate_winter(data)
    elif report_mode == 'S':
        return calculate_summer(data)
    else:
        return first_hit_date(data)

def save_excel(date_today, report_mode, data, report_content):
    # TODO: add date to file name
    with pd.ExcelWriter("gas_storage_report.xlsx", engine="xlsxwriter") as writer:
        data.to_excel(writer, sheet_name="api_data", index=False)

        if report_mode == 'W':
            report_content.to_excel(writer, sheet_name="calculated_results", index=True)    # TODO: separate constraints
        elif report_mode == 'S':
            summer_data, policy_dates = report_content
            policy_dates.to_excel(writer, sheet_name="calculated_results", startrow=0, index=False)
            summer_data.to_excel(writer, sheet_name="results", startrow=len(policy_dates) + 3, index=False)


def save_pdf(season_mode, report_content, issues):  # TODO: make pdf
#    for issue in issues:
#        print("WARNING:", issue)
    return


def main():
    today = date.today()
    season_mode = decide_season_mode(today)
    data_json = fetch_data(today, season_mode)
    data = to_dataframe(data_json)
    issues = check_data(data, today)
    goal_reached = check_policy_goal(data)
    # TODO: finish main()
    return


if __name__ == "__main__":
    main()
