import psycopg2
import os
import json
import google.generativeai as genai
import hashlib
import secrets

from psycopg2.extras import RealDictCursor
from datetime import date
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from calendar import monthrange

from app.db import get_conn
from app.custom_json import DecimalEncoder

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_headers=['*'],
    allow_methods=['*'],
)

agg_cache = {}

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
    if bool(os.getenv('CHECK_AUTH')) and not check_user_loggedin(request):
        return {'status':'unauthenticated'}
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

def get_ai_summary(data):
    # implement Gemini API summary generation


    # read API key from environment variable
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # use free Gemini model
    # model = genai.GenerativeModel("gemini-2.5-flash-lite")
    model = genai.GenerativeModel("gemma-3-27b-it")

    # construct prompt with strict word limit
    prompt = f"""
    You are an analytics assistant for an Aadhaar operations dashboard.

    You will be given spike analysis results derived from UIDAI datasets.
    The data includes:
    - Spike type (Enrollment / Demographic Update / Biometric Update)
    - State, District, and PIN code
    - Time period of spike
    - Whether the spike is sudden, recurring, or seasonal
    - Age group involved (if available)

    Your task:
    Generate SHORT, ACTIONABLE, and OPERATIONAL predictions in BULLET POINTS.

    Rules:
    - Do NOT explain the data.
    - Do NOT use technical or AI-heavy language.
    - Each bullet must be a clear recommendation or prediction.
    - Keep each bullet to 1–2 lines maximum.
    - Focus on staffing, centers, devices, or workflow decisions.

    Output format (strict):
    - <Actionable prediction 1>
    - <Actionable prediction 2>
    - <Actionable prediction 3>

    Examples of expected output:
    - Deploy temporary enrollment camps for the next 30 days in this PIN.
    - Prioritize update-only counters instead of expanding enrollment capacity.
    - Schedule biometric device recalibration and operator retraining.
    - Increase staffing during peak weeks to manage recurring spikes.
    - Shift resources from low-activity districts to high-pressure areas.

    Now generate predictions based ONLY on the spike patterns provided.

    DATA:
    {json.dumps(data, indent=2, cls=DecimalEncoder)}
    """

    try:
        # generate content using Gemini
        response = model.generate_content(prompt)

        # safely extract text response
        summary = response.text.strip() if response and response.text else ""

        return summary[:1200]  # hard cap to stay well under 200 words

    except Exception as e:
        # graceful failure handling\
        print("Gemini error:", e)
        return "AI summary could not be generated at this time."


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
    if bool(os.getenv('CHECK_AUTH')) and not check_user_loggedin(request):
        return {'status':'unauthenticated'}
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

        # initialize response dict early
        response_data = {"status": "ok", "data": {}}

        # =========================
        # monthly -> MONTHLY AGG
        # =========================
        if type == "monthly":

            if not year:
                return {"status": "incomplete request"}

            def fetch_yearly(table, columns):
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
                    c: {"x": list(range(1, 13)), "y": data[c]}
                    for c in columns
                }

            # populate dict instead of returning
            response_data["data"] = {
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
            }

        # =========================
        # daily -> DAY OF MONTH
        # =========================
        elif type == "daily":

            if not (year and month):
                return {"status": "incomplete request"}

            days_in_month = monthrange(int(year), int(month))[1]

            def fetch_daily(table, columns):
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
                    c: {"x": x_axis, "y": data[c]}
                    for c in columns
                }

            # populate dict instead of returning
            response_data["data"] = {
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
            }

        else:
            return {"status": "invalid request"}

        # generate AI summary after full data is ready
        response_data["summary"] = get_ai_summary(response_data["data"])

        # single return point
        return response_data

    finally:
        cur.close()
        conn.close()


@app.get("/aggregate")
def get_aggregate(
    request: Request,
    response: Response,
    type: str | None = None,
    state: str | None = None,
    year: str | None = None,
    month: str | None = None,
):
    if bool(os.getenv('CHECK_AUTH')) and not check_user_loggedin(request):
        return {'status':'unauthenticated'}
    # basic validation
    if type not in {"yearly", "monthly"}:
        return {"status": "invalid request"}

    if state == None:
        today_data = None
        if type == 'yearly':
            data = agg_cache.get(('yearly', year))  
            if data is not None:
                today_data = data.get(date.today)
                if today_data == None:
                    del agg_cache[('yearly', year)] 
            
        else:            
            data = agg_cache.get(('monthly', year, month))
            if data is not None:
                today_data = data.get(date.today)
                if today_data == None:
                    del agg_cache[('monthly', year, month)]
                    
        if today_data is not None:
            return today_data     


    conn = get_conn()
    cur = conn.cursor()

    try:
        # ==========================================================
        # CASE 1: STATE IS PROVIDED
        # ==========================================================
        if state:

            # fetch districts sorted alphabetically
            cur.execute("""
                SELECT DISTINCT district
                FROM enrollment_data
                WHERE state = %s
                ORDER BY district
            """, (state,))
            districts = [r[0] for r in cur.fetchall()]

            if not districts:
                return {"status": "ok", "data": {}}

            # ---------------- YEARLY AGG ----------------
            if type == "yearly":
                if not year:
                    return {"status": "incomplete request"}

                def agg_yearly(table, col):
                    # aggregate per district
                    cur.execute(f"""
                        SELECT district, SUM({col})
                        FROM {table}
                        WHERE state = %s
                          AND EXTRACT(YEAR FROM date) = %s
                        GROUP BY district
                    """, (state, int(year)))

                    rows = dict(cur.fetchall())
                    return [rows.get(d, 0) for d in districts]

                return {
                    "status": "ok",
                    "data": {
                        "districts": districts,
                        "enrollment": agg_yearly("enrollment_data", "age_0_5 + age_5_17 + age_18_greater"),
                        "biometric": agg_yearly("biometric_data", "bio_age_5_17 + bio_age_17_"),
                        "demographic": agg_yearly("demographic_data", "demo_age_5_17 + demo_age_17_"),
                    },
                }

            # ---------------- MONTHLY AGG ----------------
            if type == "monthly":
                if not (year and month):
                    return {"status": "incomplete request"}

                def agg_monthly(table, col):
                    # aggregate per district for given month
                    cur.execute(f"""
                        SELECT district, SUM({col})
                        FROM {table}
                        WHERE state = %s
                          AND EXTRACT(YEAR FROM date) = %s
                          AND EXTRACT(MONTH FROM date) = %s
                        GROUP BY district
                    """, (state, int(year), int(month)))

                    rows = dict(cur.fetchall())
                    return [rows.get(d, 0) for d in districts]

                return {
                    "status": "ok",
                    "data": {
                        "districts": districts,
                        "enrollment": agg_monthly("enrollment_data", "age_0_5 + age_5_17 + age_18_greater"),
                        "biometric": agg_monthly("biometric_data", "bio_age_5_17 + bio_age_17_"),
                        "demographic": agg_monthly("demographic_data", "demo_age_5_17 + demo_age_17_"),
                    },
                }

        # ==========================================================
        # CASE 2: STATE IS NOT PROVIDED (ALL STATES)
        # ==========================================================
        else:
            # fetch all states
            cur.execute("""
                SELECT DISTINCT state
                FROM enrollment_data
                ORDER BY state
            """)
            states = [r[0] for r in cur.fetchall()]

            if not states:
                return {"status": "ok", "data": {}}

            def agg_all_states(table, col, extra_where="", extra_params=()):
                cur.execute(f"""
                    SELECT state, SUM({col})
                    FROM {table}
                    {extra_where}
                    GROUP BY state
                """, extra_params)

                rows = dict(cur.fetchall())
                return [rows.get(s, 0) for s in states]

            # ---------------- YEARLY ----------------
            if type == "yearly":
                if not year:
                    return {"status": "incomplete request"}

                where = "WHERE EXTRACT(YEAR FROM date) = %s"
                params = (int(year),)

                data = {
                    "status": "ok",
                    "data": {
                        "states": states,
                        "enrollment": agg_all_states(
                            "enrollment_data",
                            "age_0_5 + age_5_17 + age_18_greater",
                            where,
                            params,
                        ),
                        "biometric": agg_all_states(
                            "biometric_data",
                            "bio_age_5_17 + bio_age_17_",
                            where,
                            params,
                        ),
                        "demographic": agg_all_states(
                            "demographic_data",
                            "demo_age_5_17 + demo_age_17_",
                            where,
                            params,
                        ),
                    },
                }


                agg_cache[('yearly', year)] = {date.today: data}
                print('Set cache yearly')
                return data

            # ---------------- MONTHLY ----------------
            if type == "monthly":
                if not (year and month):
                    return {"status": "incomplete request"}

                where = """
                    WHERE EXTRACT(YEAR FROM date) = %s
                      AND EXTRACT(MONTH FROM date) = %s
                """
                params = (int(year), int(month))

                data = {
                    "status": "ok",
                    "data": {
                        "states": states,
                        "enrollment": agg_all_states(
                            "enrollment_data",
                            "age_0_5 + age_5_17 + age_18_greater",
                            where,
                            params,
                        ),
                        "biometric": agg_all_states(
                            "biometric_data",
                            "bio_age_5_17 + bio_age_17_",
                            where,
                            params,
                        ),
                        "demographic": agg_all_states(
                            "demographic_data",
                            "demo_age_5_17 + demo_age_17_",
                            where,
                            params,
                        ),
                    },
                }
                agg_cache[('monthly', year, month)] = {date.today : data}
                return data

        return {"status": "invalid request"}

    finally:
        # guaranteed cleanup
        cur.close()
        conn.close()

@app.post("/upload_csv")
async def upload_csv(
    request: Request,
    response: Response,
    type: str,
    file: UploadFile = File(...)
):
    if bool(os.getenv('CHECK_AUTH')) and not check_user_loggedin(request):
        return {'status':'unauthenticated'}
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
        "biometric": {"bio_age_5_17", "bio_age_17_"},
        "demographic": {"demo_age_5_17", "demo_age_17_"},
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



@app.post("/login")
def login(request: Request, response: Response, username: str, password: str):
    if check_user_loggedin(request):
        return {'status':'already loggedin'}


    password_hash = hashlib.sha256(password.encode()).hexdigest()

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # validate credentials
        cur.execute(
            """
            SELECT username
            FROM ud_users
            WHERE username = %s AND password_hash = %s
            """,
            (username, password_hash)
        )
        user = cur.fetchone()

        if not user:
            return {"status": "invalid credentials"}

        # generate session token
        token = secrets.token_urlsafe(32)
        token_hash = hash_token(token)

        # store hashed token in memory
        sessions[token_hash] = username

        # set raw token in cookie
        response.set_cookie(
            key="token",
            value=token,
            httponly=True
        )

        return {"status": "ok"}

    finally:
        cur.close()
        conn.close()

    pass

@app.post("/signup")
def signup(request: Request, response: Response, username: str, password: str):
    if check_user_loggedin(request):
        return {'status':'already loggedin'}
    
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # check if user already exists
        cur.execute(
            "SELECT 1 FROM ud_users WHERE username = %s",
            (username,)
        )
        if cur.fetchone():
            return {"status": "user already exists"}

        # hash password before storing
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        cur.execute(
            """
            INSERT INTO ud_users (username, password_hash)
            VALUES (%s, %s)
            """,
            (username, password_hash)
        )
        conn.commit()

        return {"status": "ok"}

    finally:
        cur.close()
        conn.close()
    

@app.post("/logout")
def logout(request: Request, response:Response):
    if not check_user_loggedin(request):
        return {'status':'already loggedout'}

    del sessions[hash_token(request.cookies.get('token'))]
    
    return {'status':'ok'}

sessions = {}

def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

def check_user_loggedin(request: Request):
       
    token = request.cookies.get('token')
    if token == None:
        return False
    
    token_hash = hash_token(token)
    return token_hash in sessions

