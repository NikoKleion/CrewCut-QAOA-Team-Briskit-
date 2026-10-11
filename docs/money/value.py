import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_inputs():
    with (HERE / "inputs.csv").open(newline="", encoding="utf-8") as file:
        return {row["name"]: float(row["value"]) for row in csv.DictReader(file)}


def both(value):
    return value, value


def scaled(factor, pair):
    return factor * pair[0], factor * pair[1]


def rows(v):
    year = v["outage_cost_year_low"], v["outage_cost_year_high"]
    sandy = v["sandy_cost_low"], v["sandy_cost_high"]
    mean_outage = scaled(v["years"] / v["large_outages"], year)
    run_100 = v["task_price"] + 100 * v["shot_price"]
    run_2000 = v["task_price"] + 2000 * v["shot_price"]
    crewcut = v["crewcut_100_shots_above_best"]
    published = v["published_algorithm_above_best"]
    random_plans = v["random_100_plans_above_best"]
    room_day = 480 * v["operating_room_minute"]
    return [
        ("one_percent_of_a_year", *scaled(0.01, year), "USD per year", "0.01 * outage_cost_year"),
        ("one_tenth_percent_of_a_year", *scaled(0.001, year), "USD per year", "0.001 * outage_cost_year"),
        ("one_percent_of_sandy", *scaled(0.01, sandy), "USD", "0.01 * sandy_cost"),
        ("sandy_per_customer_out", *scaled(1 / v["sandy_customers"], sandy), "USD per customer", "sandy_cost / sandy_customers"),
        ("mean_large_outage", *mean_outage, "USD", "years * outage_cost_year / large_outages"),
        ("one_percent_of_mean_large_outage", *scaled(0.01, mean_outage), "USD", "0.01 * mean_large_outage"),
        ("run_of_100_shots", *both(run_100), "USD", "task_price + 100 * shot_price"),
        ("run_of_2000_shots", *both(run_2000), "USD", "task_price + 2000 * shot_price"),
        ("hour_more_household", *both((v["household_16_hours"] - v["household_8_hours"]) / 8), "USD per hour",
         "(household_16_hours - household_8_hours) / 8"),
        ("hour_more_small_business", *both((v["small_business_16_hours"] - v["small_business_8_hours"]) / 8), "USD per hour",
         "(small_business_16_hours - small_business_8_hours) / 8"),
        ("hour_more_large_business", *both((v["large_business_16_hours"] - v["large_business_8_hours"]) / 8), "USD per hour",
         "(large_business_16_hours - large_business_8_hours) / 8"),
        ("major_event_share_of_outage_time_2022",
         *both(100 * (v["hours_out_2022"] - v["hours_out_without_major_events"]) / v["hours_out_2022"]), "percent",
         "100 * (hours_out_2022 - hours_out_without_major_events) / hours_out_2022"),
        ("crewcut_against_published_algorithm", *scaled(published - crewcut, year), "USD per year",
         "(published_algorithm_above_best - crewcut_100_shots_above_best) * outage_cost_year"),
        ("crewcut_share_of_gap_to_best", *both(100 * (1 - crewcut / published)), "percent",
         "100 * (1 - crewcut_100_shots_above_best / published_algorithm_above_best)"),
        ("crewcut_against_random_plans", *scaled(random_plans - crewcut, year), "USD per year",
         "(random_100_plans_above_best - crewcut_100_shots_above_best) * outage_cost_year"),
        ("crewcut_to_best_plan", *scaled(crewcut, year), "USD per year", "crewcut_100_shots_above_best * outage_cost_year"),
        ("classical_lead_over_crewcut", *scaled(v["classical_lead_at_200_plans"], year), "USD per year",
         "classical_lead_at_200_plans * outage_cost_year"),
        ("one_percent_of_flight_delays", *both(0.01 * v["flight_delay_cost_2007"]), "USD per year", "0.01 * flight_delay_cost_2007"),
        ("operating_room_day", *both(room_day), "USD", "480 * operating_room_minute"),
        ("one_percent_of_operating_room_day", *both(0.01 * room_day), "USD", "0.01 * operating_room_day"),
    ]


def main():
    with (HERE / "value.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["name", "low", "high", "unit", "formula"])
        for name, low, high, unit, formula in rows(read_inputs()):
            writer.writerow([name, f"{low:.2f}", f"{high:.2f}", unit, formula])


if __name__ == "__main__":
    main()
