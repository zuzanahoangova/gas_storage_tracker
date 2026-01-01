import pandas as pd
import numpy as np
import os
import requests
from datetime import date, timedelta

today = date.today()


# pick op mode (summer, winter, transition)
def season_mode(date_today):
    m, d = date_today.month, date_today.day
    if (m == 3 and d == 31) or (4 <= m <= 9) or (m == 10 and d == 1):
        return 1    # summer mode march 31 - oct 1
    elif (m == 12 and d >= 2) or (m in [1, 2]) or (m == 3 and d <= 30):
        return 2    # winter mode dec 2 - march 30
    else:
        return 0    # transition oct 2 - dec 1


# fetch data from agsi
def fetch_data(mode, dunno):  # TODO: learn API
    # TODO: save in current folder as default but implement choose
    return


# load data from same folder
def load_data(filename):
    data = pd.read_json(filename)
    return data


# check for missing data
def check_data(data):   # TODO: do a data check (correct date, no missing days etc.)
    return
# give a warning window with missing data and let decide if proceed or not.
# or maybe just proceed anyways and just list what data is missing?


# clean data
def extract_data(data):
    data = data.iloc[:14]
    st_spec = pd.DataFrame({
        "gas_day": [data.iloc[0, 2]],               # current day as string
        "tech_capacity": [data.iloc[0, 10]],        # in TWh
        "current_storage": [data.iloc[0, 3]],       # in TWh
    })
    st_flow = data.iloc[:, [8, 9]]
    st_flow.columns = ["injection", "withdrawal"]   # in GWh
    return st_spec, st_flow


def calculate_summer(st_spec, st_flow):
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
    injection14 = st_flow["injection"].mean()
    withdrawal7 = st_flow["withdrawal"].iloc[:7].mean()
    withdrawal14 = st_flow["withdrawal"].mean()

    # all following flows in GWh
    # calculations assume net injection regime (injection > withdrawal)  # TODO: what if low/no injections?

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


def calculate_winter(st_spec, st_flow):
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
    withdrawal14 = st_flow["withdrawal"].mean()

    # How many days of coverage left assuming no more injections?
    days_of_cover7 = current_storage / withdrawal7
    days_of_cover14 = current_storage / withdrawal14

    if np.isinf(days_of_cover7):
        end_date7 = pd.NaT
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


def pick_season(mode, st_spec, st_flow):
    if mode == 1:
        return calculate_summer(st_spec, st_flow)
    elif mode == 2:
        return calculate_winter(st_spec, st_flow)
    else:
        return
    # TODO: logic for Sept-Dec 1
    # TODO: summer returns 2 objects, winter 1 -> resolve. do i even have to resolve it tho?
# run summer calc from march 31? fixed winter from dec 1 to march 30.
# transition window from mid-sept to dec 1 depends on 90% target.
# i might need to separate "date categorization" and "transition season determination".


def save_excel():
    return


def save_pdf():
    return


def main():
    return


if __name__ == "__main__":
    main()
