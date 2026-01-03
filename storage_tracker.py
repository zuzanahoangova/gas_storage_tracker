import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from enum import Enum
from datetime import date, datetime, timedelta, timezone
import subprocess
from pathlib import Path
from jinja2 import Template


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {font-family: Arial, Helvetica, sans-serif; font-size: 12pt;
    line-height: 1.3; letter-spacing: normal; word-spacing: normal; }
h1 { font-size: 18pt; }
h2 { font-size: 12pt; margin-top: 20px; }
table { border-collapse: collapse; margin-top: 10px; }
th, td { border: none; padding: 4px 6px; }
td { text-align: right; }
th { text-align: left; font-weight: bold; }
p, td, th, strong {
    text-justify: none;
    word-spacing: normal;
    letter-spacing: normal;
}
.left { text-align: left; }
.warning { color: #b00020; }
</style>
</head>
<body>

<h1>Gas Storage Report – Czech Republic</h1>

<p>
Generated: {{ generated }} (UTC)<br>
Dataset from {{ data_from }} to {{ data_to }}
</p>

{% if warnings %}
<h2>Data Warnings</h2>
<ul>
{% for w in warnings %}
<li class="warning">{{ w }}</li>
{% endfor %}
</ul>
{% endif %}

{% for line in summary %}
<p>{{ line }}</p>
{% endfor %}

{% if table %}
<p>Outputs based on 7- and 14-day averages:</p>
{{ table | safe }}
{% endif %}

{% if verdict %}
<p><strong>{{ verdict }}</strong></p>
{% endif %}

</body>
</html>
"""


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
    end_date = date_today - timedelta(days=2)  # 2 days of data lag
    if season_mode is SeasonMode.TRANSITION:   # fetches 2 months of data for transition period, otherwise 2 weeks
        start_date = date_today - timedelta(days=63)
    else:
        start_date = date_today - timedelta(days=16)

    params = {
        "country": "CZ",
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "size": 100,
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


# correct data types
def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.columns.difference(["status", "gasDayStart", "gasDayEnd"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["gasDayStart"] = pd.to_datetime(df["gasDayStart"], errors="coerce")
    df["gasDayEnd"] = pd.to_datetime(df["gasDayEnd"], errors="coerce")
    return df


# convert json data to pandas dataframe
def create_dataframe(raw_json: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(raw_json)
    data = normalize_dtypes(data)
    return data


# check for issues in data
def check_data(data, date_today):
    issues = []
    dates = data['gasDayEnd']
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
    st_flow = data.loc[:14, ['injection', 'withdrawal']]     # in GWh
    return st_spec, st_flow


def calculate_summer(data):
    st_spec, st_flow = extract_data(data)
    current_storage = st_spec.loc[0, "current_storage"]     # in TWh
    tech_capacity = st_spec.loc[0, "tech_capacity"]         # in TWh
    gas_day = st_spec.loc[0, "gas_day"]                     # current gas day

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
    min_inj_nov1_7 = (tech_capacity * factor - current_storage) * 1000 / days_to_nov1 + withdrawal7
    min_inj_nov1_14 = (tech_capacity * factor - current_storage) * 1000 / days_to_nov1 + withdrawal14
    min_inj_dec1_7 = (tech_capacity * factor - current_storage) * 1000 / days_to_dec1 + withdrawal7
    min_inj_dec1_14 = (tech_capacity * factor - current_storage) * 1000 / days_to_dec1 + withdrawal14

    summer_data = pd.DataFrame(
        {
            "avg_7d": [injection7, withdrawal7, days_to_goal7, goal_date7,
                       min_inj_nov1_7, min_inj_dec1_7],
            "avg_14d": [injection14, withdrawal14, days_to_goal14, goal_date14,
                        min_inj_nov1_14, min_inj_dec1_14],
        },
        index=["injection", "withdrawal", "days_to_goal", "goal_hit_date", "min_inj_nov1", "min_inj_dec1"]
    )

    policy_dates = pd.DataFrame(
        {
            "nov1": [nov1, days_to_nov1],
            "dec1": [dec1, days_to_dec1],
        },
        index=["policy_deadline", "days_to_deadline"]
    )
    return summer_data, policy_dates


def calculate_winter(data):
    st_spec, st_flow = extract_data(data)
    current_storage = st_spec.loc[0, "current_storage"] * 1000  # in GWh
    gas_day = st_spec.loc[0, "gas_day"]                         # current gas day

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


# writes a new sheet with tables from 2 dataframes
def write_branch(writer, table1, table2):
    sheet_name = "calculated_results"
    # index labels not saved, extract before calling function
    table1.to_excel(writer, sheet_name=sheet_name, startrow=2, startcol=0, index=False)
    table2.to_excel(writer, sheet_name=sheet_name,
                    startrow=2, startcol=table1.shape[1] + 1, index=False)
    worksheet = writer.sheets[sheet_name]
    worksheet.write(0, 0, "Gas storage report – summary")
    worksheet.write(1, 0, "all flows in GWh")


# save input and output data to excel file
def save_excel(report_mode, data, report_content):
    now = datetime.now(timezone.utc)    # UTC for auditability
    timestamp = now.strftime("%Y-%m-%d_%H%M%S_UTC")
    filename = f"Gas_Storage_Report_{timestamp}.xlsx"
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        data.to_excel(writer, sheet_name="api_data", index=False)

        if report_mode == 'W':
            constraints = report_content[["ref"]].copy()
            constraints.insert(0, "par_name", ["end_of_winter", "days_to_eow", "max_withdrawal_eow"])
            winter_data = report_content[["avg_7d", "avg_14d"]].copy()
            winter_data.insert(0, "par_name", ["cover_end_date", "days_of_cover", "withdrawal_avg"])
            write_branch(writer, constraints, winter_data)

        elif report_mode == 'S':
            summer_data, policy_dates = report_content
            policy_dates = policy_dates.reset_index(names="par_name")  # extracts index labels
            summer_data = summer_data.reset_index(names="par_name")
            write_branch(writer, policy_dates, summer_data)


def render_pdf(html: str, output_path: str):
    tmp_html = Path("_temp_report.html")
    tmp_html.write_text(html, encoding="utf-8")
    subprocess.run(["wkhtmltopdf", "--encoding", "utf-8", str(tmp_html), output_path],check=True)
    tmp_html.unlink()


def save_pdf(report_mode, report_content, issues, data):
    # timestamps
    now = datetime.now(timezone.utc)
    timestamp_text = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_file = now.strftime("%Y-%m-%d_%H%M%S_UTC")

    # header
    data_from = data["gasDayEnd"].min().date()
    data_to = data["gasDayEnd"].max().date()

    summary = []
    table = None
    verdict = None

    # mode: summer completed
    if report_mode is None:
        hit_date = report_content
        current_fill = data.loc[0, 'full']
        summary.append(f"Policy storage threshold (90%) was reached on {hit_date.date()} or earlier.")
        summary.append(f"Current storage fill: {current_fill} % of working gas capacity.")

    # mode: winter
    elif report_mode == "W":
        winter = report_content
        current_fill = data.loc[0, "full"]

        eow = winter.loc["date", "ref"].date()
        days_to_eow = int(winter.loc["days", "ref"])
        max_withdrawal = winter.loc["withdrawal", "ref"]

        summary.append(f"Current storage fill: {current_fill} % of working gas capacity.")
        summary.append(f"Days until end of winter ({eow}): {days_to_eow}")
        summary.append(f"Maximum sustainable withdrawal rate to {eow}: {max_withdrawal:.0f} GWh/day")

        # winter data table
        table = winter[["avg_7d", "avg_14d"]].copy()

        table.loc["date"] = table.loc["date"].apply(lambda x: x.date() if pd.notna(x) else x)
        table.loc["days"] = table.loc["days"].apply(np.floor).astype("Int64")
        table.loc["withdrawal"] = table.loc["withdrawal"].apply(
            lambda x: f"{x:.3g}" if pd.notna(x) and isinstance(x, (int, float)) else x)

        table.index = ["Last cover date:", "Days of cover:", "Average withdrawal (GWh/day):"]
        table.columns = ["7-day", "14-day"]

        if min(table.loc["Last cover date:"]) >= eow:   # returns the earlier date
            verdict = "System remains on track through end of winter."
        else:
            verdict = "System does not remain covered through end of winter."

    # mode: summer active
    else:
        summer, policy = report_content
        current_fill = data.loc[0, "full"]

        sd = policy.loc["policy_deadline", "nov1"].date()
        hd = policy.loc["policy_deadline", "dec1"].date()
        days_sd = int(policy.loc["days_to_deadline", "nov1"])
        days_hd = int(policy.loc["days_to_deadline", "dec1"])

        summary.append("Policy goal: 90 % storage fill")
        summary.append(f"Current storage fill: {current_fill} % of working gas capacity.")
        summary.append(f"Soft deadline (SD): {sd} – Days to SD: {days_sd}")
        summary.append(f"Hard deadline (HD): {hd} – Days to HD: {days_hd}")

        # summer data table
        table = summer.drop(index="withdrawal").copy()

        table.loc["goal_hit_date"] = table.loc["goal_hit_date"].apply(lambda x: x.date() if pd.notna(x) else x)
        table.loc["days_to_goal"] = table.loc["days_to_goal"].apply(np.ceil).astype("Int64")
        inj_rows = ["injection", "min_inj_nov1", "min_inj_dec1"]
        table.loc[inj_rows] = table.loc[inj_rows].applymap(
            lambda x: f"{x:.3g}" if pd.notna(x) and isinstance(x, (int, float)) else x)

        table.index = [
            "Average injection (GWh/day):",
            "Days to reach 90 %:",
            "Date when 90 % reached:",
            "Min. needed injection rate to SD (GWh/day):",
            "Min. needed injection rate to HD (GWh/day):",
        ]
        table.columns = ["7-day", "14-day"]

        # policy status verdict
        days_to_90 = table.loc["Days to reach 90 %:"].max()   # selects the day closer to deadline

        if days_to_90 > days_hd:
            verdict = "Policy goal not achievable by hard deadline."
        elif days_sd < days_to_90 <= days_hd:
            verdict = "Policy goal achievable, but margin is limited."
        else:
            verdict = "Policy goal achievable with comfortable margin."

    # render
    html = Template(HTML_TEMPLATE).render(
        generated=timestamp_text,
        data_from=data_from,
        data_to=data_to,
        warnings=issues,
        summary=summary,
        table=table.to_html(border=0) if table is not None else None,
        verdict=verdict,
    )

    filename = f"Gas_Storage_Report_{timestamp_file}.pdf"
    render_pdf(html, filename)


def main():
    # test dates
    # today = date(2025, 6, 9)
    # today = date(2025, 9, 21)
    # today = date(2025, 11, 9)
    today = date.today()

    season_mode = decide_season_mode(today)
    data_json = fetch_data(today, season_mode)
    data = create_dataframe(data_json)

    goal_reached = check_policy_goal(data)
    report_mode = decide_calculation(season_mode, goal_reached)
    report_content = calculate_report_content(report_mode, data)

    issues = check_data(data, today)
    save_pdf(report_mode, report_content, issues, data)

    save_excel(report_mode, data, report_content)
    return


if __name__ == "__main__":
    main()
