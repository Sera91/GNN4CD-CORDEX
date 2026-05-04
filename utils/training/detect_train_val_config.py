from datetime import date
import random

def detect_train_val_config(args):
    """
    Normalize all train/val configuration scenarios into a unified structure.
    """

    # Case 1: explicit year lists
    if args.train_years and args.val_years:
        train_years = [int(y) for y in args.train_years.split()]
        val_years   = [int(y) for y in args.val_years.split()]
        return {
            "mode": "years_list",
            "train_years": train_years,
            "val_years": val_years
        }

    # Case 2: random sampling from a year range
    if args.first_year and args.last_year:
        all_years = list(range(int(args.first_year), int(args.last_year) + 1))
        val_years = random.sample(all_years, int(args.n_val_years))
        train_years = [y for y in all_years if y not in val_years]
        return {
            "mode": "random_years",
            "train_years": train_years,
            "val_years": val_years
        }

    # Case 3: date range + validation year
    train_start = date(
        int(args.train_year_start),
        int(args.train_month_start),
        int(args.train_day_start)
    )
    train_end = date(
        int(args.train_year_end),
        int(args.train_month_end),
        int(args.train_day_end)
    )

    return {
        "mode": "date_range",
        "train_start": train_start,
        "train_end": train_end,
        "validation_year": int(args.validation_year)
    }
