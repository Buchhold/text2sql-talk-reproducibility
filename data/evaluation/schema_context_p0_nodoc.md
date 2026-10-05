# Schema Context — `ecom-v1` (P0: no documentation)

Prompt input for `condition=P0`. Bare DDL only — no comments, no glossary,
no column-trap documentation, no output conventions. This is the harder
baseline: it represents a newly connected legacy system with no curated
schema documentation available.

Compare against `schema_context.md` (`condition=P1`, curated context).
Everything else — model, prompt template, question set, execution and
scoring logic — stays identical between P0 and P1. Only this file differs.

BigQuery dialect. Dataset alias used below: `ecom_v1` (the deployed dataset
is `ecom_v1` in project `example-project`, region `europe-west3` — the
runner qualifies table references at execution time, not this document).

**Snapshot date: `2026-06-30`.** This is a fixed benchmark constant, not
documentation — it must be given even in the undocumented condition,
otherwise the run measures whether the model can guess the reference date
instead of measuring the effect of missing schema documentation.

---

```sql
CREATE TABLE ecom_v1.dim_date (
  date_key            DATE    NOT NULL,
  year                INT64   NOT NULL,
  quarter             INT64   NOT NULL,
  month               INT64   NOT NULL,
  day_of_month        INT64   NOT NULL,
  day_of_week         INT64   NOT NULL,
  iso_week            INT64   NOT NULL,
  is_weekend          BOOL    NOT NULL,
  is_public_holiday   BOOL    NOT NULL
);

CREATE TABLE ecom_v1.categories (
  category_id         INT64   NOT NULL,
  category_name       STRING  NOT NULL,
  parent_category_id  INT64
);

CREATE TABLE ecom_v1.products (
  product_id          INT64   NOT NULL,
  sku                 STRING  NOT NULL,
  product_name        STRING  NOT NULL,
  category_id         INT64   NOT NULL,
  list_price          NUMERIC NOT NULL,
  purchase_price      NUMERIC NOT NULL,
  launched_on         DATE    NOT NULL,
  discontinued_on     DATE
);

CREATE TABLE ecom_v1.customers (
  customer_id         INT64   NOT NULL,
  customer_number     STRING  NOT NULL,
  signup_date         DATE    NOT NULL,
  country_code        STRING  NOT NULL,
  postal_code         STRING,
  segment             STRING  NOT NULL,
  newsletter_optin    BOOL    NOT NULL,
  deleted_at          TIMESTAMP
);

CREATE TABLE ecom_v1.channels (
  channel_id          INT64   NOT NULL,
  channel_name        STRING  NOT NULL,
  channel_type        STRING  NOT NULL,
  commission_rate     NUMERIC NOT NULL
);

CREATE TABLE ecom_v1.orders (
  order_id            INT64   NOT NULL,
  customer_id         INT64   NOT NULL,
  channel_id          INT64   NOT NULL,
  order_date          DATE    NOT NULL,
  ship_date           DATE,
  order_status        STRING  NOT NULL,
  shipping_cost       NUMERIC NOT NULL,
  payment_method      STRING  NOT NULL
);

CREATE TABLE ecom_v1.order_items (
  order_item_id       INT64   NOT NULL,
  order_id            INT64   NOT NULL,
  product_id          INT64   NOT NULL,
  quantity            INT64   NOT NULL,
  unit_price_gross    NUMERIC NOT NULL,
  discount_amount     NUMERIC NOT NULL,
  tax_rate            NUMERIC NOT NULL
);

CREATE TABLE ecom_v1.returns (
  return_id           INT64   NOT NULL,
  order_item_id       INT64   NOT NULL,
  return_date         DATE    NOT NULL,
  quantity_returned   INT64   NOT NULL,
  reason_code         STRING  NOT NULL,
  refund_amount       NUMERIC NOT NULL
);
```
