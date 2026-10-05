# Schema Context — `ecom-v1` (P1: glossary, no schema comments)

Prompt input for `condition=P1`. Bare DDL (identical to `schema_context_p0_nodoc.md`
— no inline column comments, no column-trap table derived from schema structure)
plus the full house terminology glossary and output conventions.

Represents the realistic case where business definitions are documented
independently of the schema itself (e.g. in a wiki, a data dictionary, or
tribal knowledge written down by the business/data team) — regardless of
whether the underlying DDL happens to carry helpful comments. Whether a
legacy warehouse has commented DDL is incidental to the system's history;
whether house terminology is written down anywhere is a separate, and for
many CID engagements more consequential, question.

Compare against `schema_context_p0_nodoc.md` (`condition=P0`, nothing) and
`schema_context.md` (`condition=P2`, glossary + commented DDL + column
traps). Model, prompt template, question set, execution and scoring logic
stay identical across P0/P1/P2 — only this file differs.

BigQuery dialect. Dataset alias used below: `ecom_v1` (the deployed dataset
is `ecom_v1` in project `example-project`, region `europe-west3` — the
runner qualifies table references at execution time, not this document).

**Snapshot date: `2026-06-30`.** This is a fixed benchmark constant, not
documentation — it must be given in every condition, otherwise a run
measures whether the model can guess the reference date instead of
measuring the effect of missing context.

---

## Tables

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
  list_price           NUMERIC NOT NULL,
  purchase_price       NUMERIC NOT NULL,
  launched_on          DATE    NOT NULL,
  discontinued_on      DATE
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

---

## House terminology (authoritative — not derivable from column names)

**Kunde (customer), ungerichtet:** a bare "Kunden"/"Kundschaft" question (no
qualifier such as "aktiv") refers to all rows in `customers`, including
GDPR-deleted accounts (`deleted_at IS NOT NULL`). Do not apply the "Aktiver
Kunde" filter below unless the question explicitly asks for active
customers or otherwise references activity/account status.

**Aktiver Kunde (active customer):** a customer counts as active if both
conditions hold: at least one order with `order_status != 'cancelled'` in
the 90 days up to and including `SNAPSHOT_DATE` (i.e. `order_date BETWEEN
DATE_SUB(SNAPSHOT_DATE, INTERVAL 90 DAY) AND SNAPSHOT_DATE`), AND
`customers.deleted_at IS NULL`. There is no `is_active` flag. Common
mistakes: counting cancelled orders, or not excluding GDPR-deleted accounts.

**Netto-Umsatz (net revenue):**
`SUM(quantity * unit_price_gross - discount_amount) / (1 + tax_rate)` per
line item. Excludes cancelled orders (`order_status = 'cancelled'`)
entirely. Excludes shipping cost (`orders.shipping_cost` is never part of
revenue). VAT (`tax_rate`, 0.19 or 0.07) must be divided out of the gross
price; discounts are subtracted before dividing out VAT. By default this
does NOT subtract refunds — only subtract `refund_amount` of associated
returns when the question explicitly signals that returns must be
accounted for (e.g. "nach Abzug von Retouren", "nach allen Abzügen").

**Wiederkäufer (repeat customer):** a customer with non-cancelled orders in
at least two distinct calendar months (`FORMAT_DATE('%Y-%m', order_date)`
differs across at least two orders). Two orders on the same day, or in the
same month, do not make a repeat customer by house definition.

**Deckungsbeitrag (contribution margin):** Netto-Umsatz minus `quantity *
purchase_price`, minus commission (`commission_rate` applied to the gross
line value, only when `channel_type = 'partner'`). Requires a correct
net-revenue calculation as a prerequisite.

---

## Output conventions

- Return category, product, customer, and channel **names**, not internal
  IDs, unless the question explicitly asks for an ID.
- Return only the columns the question asks for. Do not add explanatory or
  derived columns beyond what is requested.
- Write plain BigQuery Standard SQL. No comments, no markdown fences, no
  trailing semicolon requirement — a single query per answer.
