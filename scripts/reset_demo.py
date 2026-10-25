"""Replace only FreshMart's five demo tables; never delete database files."""

import logging
import sqlite3

from scripts.generate_data import argument_parser, generate_data

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argument_parser(__doc__)
    parser.add_argument(
        "--yes", action="store_true",
        help="Confirm replacement of all demo data, including newly submitted cases",
    )
    args = parser.parse_args()
    if not args.yes:
        parser.error("Reset requires --yes; it removes all existing demo complaints.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        generate_data(
            args.db, seed=args.seed, end_date=args.end_date, reset=True
        )
    except (ValueError, sqlite3.Error, OSError) as exc:
        LOGGER.error("Demo reset failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
