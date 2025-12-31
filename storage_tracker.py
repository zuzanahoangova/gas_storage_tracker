import pandas as pd
import numpy as np
import os


# fetch data from agsi
def fetch_data(dunno):  # TODO: learn API
    # TODO: save in current folder as default but implement choose
    return


# load data from same folder
def load_data(filename):
    data = pd.read_csv(filename, delimiter=";", nrows=14)  # TODO: change file type later, filename as string
    return data


# clean data
def extract_data(data):  # TODO: do a data check (correct date, no missing days etc.)
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
    # TODO: implement 10% wing room later
    nov1 = pd.Timestamp(gas_day.year, 11, 1)  # nov 1: anchor date, old policy 2022-2025
    dec1 = pd.Timestamp(gas_day.year, 12, 1)  # dec 1: hard deadline, new policy after april 2025
    days_to_nov1 = (nov1 - gas_day).days
    days_to_dec1 = (dec1 - gas_day).days

    injection7 = st_flow["injection"].iloc[:7].mean()
    injection14 = st_flow["injection"].mean()
    withdrawal7 = st_flow["withdrawal"].iloc[:7].mean()
    withdrawal14 = st_flow["withdrawal"].mean()

    # all following flows in GWh
    # calculations assume net injection regime (injection > withdrawal)
    days_to_goal7 = (tech_capacity * factor - current_storage) * 1000 / (injection7 - withdrawal7)
    days_to_goal14 = (tech_capacity * factor - current_storage) * 1000 / (injection14 - withdrawal14)
    goal_date7 = gas_day + pd.Timedelta(days=np.ceil(days_to_goal7))
    goal_date14 = gas_day + pd.Timedelta(days=np.ceil(days_to_goal14))

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

    days_to_eow = (eow - gas_day).days  # TODO: handle zero days_to_eow / zero withdrawal edge cases
    max_withdrawal_eow = current_storage / days_to_eow
    withdrawal7 = st_flow["withdrawal"].iloc[:7].mean()
    withdrawal14 = st_flow["withdrawal"].mean()
    days_of_cover7 = current_storage / withdrawal7
    days_of_cover14 = current_storage / withdrawal14
    end_date7 = gas_day + pd.Timedelta(days=np.floor(days_of_cover7))
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


def save_excel():
    return


def save_pdf():
    return


def main():
    return


if __name__ == "__main__":
    main()
