import psycopg2
from fastapi import FastAPI, Request, Response, UploadFile, File
from app.db import get_conn
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['*'],
)


@app.get("/filter")
def get_filter(
    request: Request,
    response: Response,
    type: str,
    state: str | None = None,
    district: str | None = None,
    pincode: str | None = None,
    year: str | None = None,
):

    # normalize empty strings
    state = state or None
    district = district or None
    pincode = pincode or None
    year = year or None

    if not type:
        return {"status": "invalid request"}

    conn = get_conn()
    cur = conn.cursor()

    try:
        if type == "state":
            # DISTINCT states from DB
            cur.execute("""
                SELECT DISTINCT state
                FROM enrollment_data
                ORDER BY state
            """)
            return {"status": "ok", "data": [r[0] for r in cur.fetchall()]}

        if type == "district":
            if not state:
                return {"status": "incomplete request"}

            cur.execute("""
                SELECT DISTINCT district
                FROM enrollment_data
                WHERE state = %s
                ORDER BY district
            """, (state,))
            return {"status": "ok", "data": [r[0] for r in cur.fetchall()]}

        if type == "pincode":
            if not state or not district:
                return {"status": "incomplete request"}

            cur.execute("""
                SELECT DISTINCT pincode
                FROM enrollment_data
                WHERE state = %s AND district = %s
                ORDER BY pincode
            """, (state, district))
            return {"status": "ok", "data": [r[0] for r in cur.fetchall()]}

        if type == "year":
            if not state:
                return {"status": "incomplete request"}

            query = """
                SELECT DISTINCT EXTRACT(YEAR FROM date)::int
                FROM enrollment_data
                WHERE state = %s
            """
            params = [state]

            if district:
                query += " AND district = %s"
                params.append(district)

            if pincode:
                query += " AND pincode = %s"
                params.append(pincode)

            query += " ORDER BY 1"

            cur.execute(query, tuple(params))
            return {"status": "ok", "data": [r[0] for r in cur.fetchall()]}

        if type == "month":
            if not state or not year:
                return {"status": "incomplete request"}

            query = """
                SELECT DISTINCT
                    EXTRACT(MONTH FROM date) AS month_num,
                    TO_CHAR(date, 'Month') AS month_name
                FROM enrollment_data
                WHERE state = %s
                AND EXTRACT(YEAR FROM date) = %s
            """
            params = [state, int(year)]

            if district:
                query += " AND district = %s"
                params.append(district)

            if pincode:
                query += " AND pincode = %s"
                params.append(pincode)

            query += " ORDER BY month_num"

            cur.execute(query, tuple(params))

            # discard month_num, return ordered names only
            return {
                "status": "ok",
                "data": [r[1].strip() for r in cur.fetchall()]
            }

        return {"status": "invalid request"}

    finally:
        # ensure DB resources are closed
        cur.close()
        conn.close()


from calendar import monthrange
from datetime import timedelta

@app.get("/data")
def get_data(
    request: Request,
    response: Response,
    type: str,
    state: str | None = None,
    district: str | None = None,
    pincode: str | None = None,
    year: str | None = None,
    month: str | None = None,
):

    # normalize empty strings
    state = state or None
    district = district or None
    pincode = pincode or None
    year = year or None
    month = month or None

    if not state:
        return {"status": "incomplete request"}

    conn = get_conn()
    cur = conn.cursor()

    try:
        # build common filters once
        filters = ["state = %s"]
        params = [state]

        if district:
            filters.append("district = %s")
            params.append(district)

        if pincode:
            filters.append("pincode = %s")
            params.append(pincode)

        where_clause = " AND ".join(filters)

        # =========================
        # monthly -> MONTHLY AGG
        # =========================
        if type == "monthly":

            if not year:
                return {"status": "incomplete request"}

            def fetch_yearly(table, columns):
                # monthly aggregation with zero-fill
                query = f"""
                    SELECT
                        EXTRACT(MONTH FROM date)::int AS m,
                        {", ".join(f"SUM({c})" for c in columns)}
                    FROM {table}
                    WHERE {where_clause}
                      AND EXTRACT(YEAR FROM date) = %s
                    GROUP BY m
                    ORDER BY m
                """

                cur.execute(query, tuple(params + [int(year)]))
                rows = cur.fetchall()

                data = {c: [0] * 12 for c in columns}

                for r in rows:
                    month_idx = r[0] - 1
                    for i, c in enumerate(columns):
                        data[c][month_idx] = r[i + 1] or 0

                return {
                    c: {
                        "x": list(range(1, 13)),
                        "y": data[c],
                    }
                    for c in columns
                }

            return {
                "status": "ok",
                "data": {
                    "enrollment": fetch_yearly(
                        "enrollment_data",
                        ["age_0_5", "age_5_17", "age_18_greater"],
                    ),
                    "biometric": fetch_yearly(
                        "biometric_data",
                        ["bio_age_5_17", "bio_age_17_"],
                    ),
                    "demographic": fetch_yearly(
                        "demographic_data",
                        ["demo_age_5_17", "demo_age_17_"],
                    ),
                },
            }

        # =========================
        # daily -> DAY OF MONTH
        # =========================
        if type == "daily":

            if not (year and month):
                return {"status": "incomplete request"}

            days_in_month = monthrange(int(year), int(month))[1]

            def fetch_daily(table, columns):
                # daily aggregation by day-of-month
                query = f"""
                    SELECT
                        EXTRACT(DAY FROM date)::int AS d,
                        {", ".join(f"SUM({c})" for c in columns)}
                    FROM {table}
                    WHERE {where_clause}
                      AND EXTRACT(YEAR FROM date) = %s
                      AND EXTRACT(MONTH FROM date) = %s
                    GROUP BY d
                    ORDER BY d
                """

                cur.execute(
                    query,
                    tuple(params + [int(year), int(month)])
                )
                rows = cur.fetchall()

                data = {c: [0] * days_in_month for c in columns}
                x_axis = list(range(1, days_in_month + 1))

                for r in rows:
                    day_idx = r[0] - 1
                    for i, c in enumerate(columns):
                        data[c][day_idx] = r[i + 1] or 0

                return {
                    c: {
                        "x": x_axis,
                        "y": data[c],
                    }
                    for c in columns
                }

            return {
                "status": "ok",
                "data": {
                    "enrollment": fetch_daily(
                        "enrollment_data",
                        ["age_0_5", "age_5_17", "age_18_greater"],
                    ),
                    "biometric": fetch_daily(
                        "biometric_data",
                        ["bio_age_5_17", "bio_age_17_"],
                    ),
                    "demographic": fetch_daily(
                        "demographic_data",
                        ["demo_age_5_17", "demo_age_17_"],
                    ),
                },
            }

        return {"status": "invalid request"}

    finally:
        # guaranteed DB cleanup
        cur.close()
        conn.close()

@app.post("/upload_csv")
async def upload_csv(
    request: Request,
    response: Response,
    type: str,
    file: UploadFile = File(...)
):
    import pandas as pd

    # validate file extension
    if not file.filename.lower().endswith(".csv"):
        return {"status": "invalid file type"}

    # validate type parameter
    if type not in {"enrollment", "biometric", "demographic"}:
        return {"status": "invalid type"}

    # ensure upload directory exists
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename

    # save uploaded file to disk
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # read CSV using pandas
    df = pd.read_csv(file_path)

    # define required columns
    common_cols = {"date", "state", "district", "pincode"}

    type_columns = {
        "enrollment": {"age_0_5", "age_5_17", "age_18_greater"},
        "biometric": {"bio_age_0_5", "bio_age_17_"},
        "demographic": {"demo_age_0_5", "demo_age_17_"},
    }

    required_cols = common_cols | type_columns[type]

    # check for missing columns
    missing = required_cols - set(df.columns)
    if missing:
        return {"status": "missing column"}

    # convert age columns from string to int
    age_cols = list(type_columns[type])
    for col in age_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows where age conversion failed
    df = df.dropna(subset=age_cols)

    # cast age columns to int64 (Postgres int8)
    df[age_cols] = df[age_cols].astype("int64")

    # clean dataframe
    df = df.dropna()
    df = df.drop_duplicates()

    # parse date column and sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # insert cleaned data into database
    conn = get_conn()
    cur = conn.cursor()

    try:
        table_map = {
            "enrollment": "enrollment_data",
            "biometric": "biometric_data",
            "demographic": "demographic_data",
        }

        table = table_map[type]
        cols = list(required_cols)
        placeholders = ", ".join(["%s"] * len(cols))

        query = f"""
            INSERT INTO {table} ({", ".join(cols)})
            VALUES ({placeholders})
        """

        cur.executemany(
            query,
            df[cols].itertuples(index=False, name=None)
        )

        conn.commit()

    finally:
        cur.close()
        conn.close()

    return {"status": "ok"}

