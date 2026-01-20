# Aadhaar Analytics Backend

This project is a **FastAPI-based backend service** designed for analyzing Aadhaar enrollment, biometric, and demographic update data. It provides structured APIs for filtering, aggregation, visualization-ready datasets, CSV ingestion, authentication, and AI-powered operational insights.

---

## Features

### Authentication

* Username/password-based login and signup
* Session-based authentication using HTTP-only cookies
* In-memory session storage
* Protected APIs (authentication required)

### Data Ingestion

* Upload CSV files for:

  * Enrollment data
  * Biometric updates
  * Demographic updates
* Automatic validation, cleaning, deduplication, and type conversion
* Bulk insertion into PostgreSQL

### Filtering APIs

* Fetch distinct values for:

  * States
  * Districts
  * PIN codes
  * Years
  * Months
* Hierarchical filtering (state → district → pincode → time)

### Data Analytics APIs

* **Monthly analytics** (per year)
* **Daily analytics** (per month)
* Outputs are chart-ready (`x` and `y` arrays)
* Separate breakdowns for:

  * Enrollment
  * Biometric updates
  * Demographic updates
* Age-group based aggregation

### Aggregation APIs

* State-wise and district-wise aggregations
* Yearly and monthly summaries
* Optimized in-memory caching for national-level aggregates

### AI Insights

* Uses Google Gemini (Gemma model)
* Generates short, actionable operational recommendations
* Focused on:

  * Staffing
  * Resource allocation
  * Center operations
* Strictly non-technical, decision-oriented output

---

## Tech Stack

* **Backend Framework:** FastAPI
* **Database:** PostgreSQL
* **ORM/Driver:** psycopg2
* **Data Processing:** Pandas
* **AI Integration:** Google Generative AI (Gemini)
* **Auth:** Cookie-based sessions
* **Caching:** In-memory Python dictionary

---

## Environment Variables

Create a `.env` file or export the following:

```bash
GEMINI_API_KEY=your_gemini_api_key
```

Database credentials are expected to be handled inside `app.db.get_conn()`.

---

## Database Tables (Expected)

### `ud_users`

| Column        | Type |
| ------------- | ---- |
| username      | text |
| password_hash | text |

### `enrollment_data`

* date
* state
* district
* pincode
* age_0_5
* age_5_17
* age_18_greater

### `biometric_data`

* date
* state
* district
* pincode
* bio_age_5_17
* bio_age_17_

### `demographic_data`

* date
* state
* district
* pincode
* demo_age_5_17
* demo_age_17_

---

## API Endpoints

### Authentication

* `POST /signup`
* `POST /login`
* `POST /logout`

### Filters

* `GET /filter`

  * Types: `state`, `district`, `pincode`, `year`, `month`

### Data Analytics

* `GET /data`

  * Types: `monthly`, `daily`
  * Returns analytics + AI summary

### Aggregations

* `GET /aggregate`

  * Types: `yearly`, `monthly`
  * Supports state-level and national-level views

### CSV Upload

* `POST /upload_csv`

  * Types: `enrollment`, `biometric`, `demographic`

---

## Caching Behavior

* National-level aggregates are cached per day
* Cache auto-invalidates daily
* Reduces repeated heavy aggregation queries

---

## Security Notes

* Passwords are hashed using SHA-256
* Session tokens are hashed before storage
* Cookies are HTTP-only
* No JWTs used

---

## Intended Use Case

This backend is designed for:

* Government dashboards
* Aadhaar operational monitoring
* Policy and planning teams
* Resource optimization and forecasting

---

## Future Improvements

* Persistent session storage (Redis / DB)
* Role-based access control
* Rate limiting
* Background jobs for heavy aggregation
* Precomputed analytics tables
