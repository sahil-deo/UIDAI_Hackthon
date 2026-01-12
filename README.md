# API Documentation

## Base URL

```
https://uidai-hackthon.onrender.com
```

All endpoints are **GET** requests and return JSON.

---

## Common Response Format

### Success

```json
{
  "status": "ok",
  "data": ...
}
```

### Errors

```json
{ "status": "invalid request" }
{ "status": "incomplete request" }
```

* **invalid request** → unsupported or missing `type`
* **incomplete request** → required query parameters not provided

---

## 1️⃣ `/filter` Endpoint

Used to **fetch available filter values** (state, district, pincode, year, month) from the database.

### Endpoint

```
GET /filter
```

### Query Parameters

| Name     | Required | Description                    |
| -------- | -------- | ------------------------------ |
| type     | Yes      | Filter type to fetch           |
| state    | No       | Required for some filter types |
| district | No       | Required for some filter types |
| pincode  | No       | Optional refinement            |
| year     | No       | Required for `month`           |

---

### Supported `type` Values

---

### 🔹 `type=state`

Returns all distinct states.

**Request**

```
/filter?type=state
```

**Response**

```json
{
  "status": "ok",
  "data": ["Karnataka", "Maharashtra", "Tamil Nadu"]
}
```

---

### 🔹 `type=district`

Requires `state`.

**Request**

```
/filter?type=district&state=Karnataka
```

**Response**

```json
{
  "status": "ok",
  "data": ["Bangalore", "Mysore"]
}
```

---

### 🔹 `type=pincode`

Requires `state` and `district`.

**Request**

```
/filter?type=pincode&state=Karnataka&district=Bangalore
```

**Response**

```json
{
  "status": "ok",
  "data": ["560001", "560002"]
}
```

---

### 🔹 `type=year`

Requires `state`.
Optionally filtered by `district` and `pincode`.

**Request**

```
/filter?type=year&state=Karnataka
```

**Response**

```json
{
  "status": "ok",
  "data": [2021, 2022, 2023]
}
```

---

### 🔹 `type=month`

Requires `state` and `year`.
Optionally filtered by `district` and `pincode`.

**Request**

```
/filter?type=month&state=Karnataka&year=2023
```

**Response**

```json
{
  "status": "ok",
  "data": ["January", "February", "March"]
}
```

Returned months are **ordered chronologically**.

---

## 2️⃣ `/data` Endpoint

Used to fetch **aggregated enrollment, biometric, and demographic data**.

### Endpoint

```
GET /data
```

### Query Parameters

| Name     | Required | Description             |
| -------- | -------- | ----------------------- |
| type     | Yes      | `monthly` or `daily`    |
| state    | Yes      | State name              |
| district | No       | Optional                |
| pincode  | No       | Optional                |
| year     | Yes      | Required for both types |
| month    | Depends  | Required for `daily`    |

---

## 📊 `type=monthly`

Returns **month-wise aggregated data** for a given year.

### Required

* `state`
* `year`

### Request

```
/data?type=monthly&state=Karnataka&year=2023
```

### Response Structure

```json
{
  "status": "ok",
  "data": {
    "enrollment": {
      "age_0_5": { "x": [1..12], "y": [...] },
      "age_5_17": { "x": [1..12], "y": [...] },
      "age_18_greater": { "x": [1..12], "y": [...] }
    },
    "biometric": {
      "bio_age_5_17": { "x": [1..12], "y": [...] },
      "bio_age_17_": { "x": [1..12], "y": [...] }
    },
    "demographic": {
      "demo_age_5_17": { "x": [1..12], "y": [...] },
      "demo_age_17_": { "x": [1..12], "y": [...] }
    }
  }
}
```

* `x` → month numbers (1–12)
* `y` → summed values
* Missing months are **zero-filled**

---

## 📅 `type=daily`

Returns **day-wise aggregated data** for a specific month.

### Required

* `state`
* `year`
* `month`

### Request

```
/data?type=daily&state=Karnataka&year=2023&month=3
```

### Response Structure

```json
{
  "status": "ok",
  "data": {
    "enrollment": {
      "age_0_5": { "x": [1..31], "y": [...] },
      "age_5_17": { "x": [1..31], "y": [...] }
    },
    "biometric": {
      "bio_age_5_17": { "x": [1..31], "y": [...] }
    },
    "demographic": {
      "demo_age_5_17": { "x": [1..31], "y": [...] }
    }
  }
}
```

* `x` → day of month
* Days without data are **zero-filled**
* Month length matches calendar (28–31)

---

## Notes for Developers

* All filters are **optional unless stated**
* Empty query strings are treated as `null`
* Responses are optimized for **chart plotting**
* Database connections are opened and closed per request
