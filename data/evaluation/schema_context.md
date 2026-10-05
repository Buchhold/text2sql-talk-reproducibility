# Schema Context — `ecom-v1`

Prompt input for all Text2SQL runs on `ecom-v1`. Identical across models and
conditions (Phase 0 protocol, section 1.7 fairness rule: "identischer
Schema-Kontext pro `condition` über alle Modelle"). Do not edit inline for a
specific run — any change is a new `schema_context_version`.

BigQuery dialect. Dataset alias used below: `ecom_v1` (the deployed dataset
is `ecom_v1` in project `example-project`, region `europe-west3` — the
runner is responsible for qualifying table references at execution time,
not this document).

**Snapshot date: `2026-06-30`.** All relative time expressions ("last 90
days", "this year", "year to date") resolve against this fixed date. Never
use `CURRENT_DATE()`.

---

## Tables

```sql
CREATE TABLE ecom_v1.dim_date (
  date_key            DATE    NOT NULL,  -- PK
  year                INT64   NOT NULL,
  quarter             INT64   NOT NULL,
  month               INT64   NOT NULL,
  day_of_month        INT64   NOT NULL,
  day_of_week         INT64   NOT NULL,  -- 1 = Monday
  iso_week            INT64   NOT NULL,
  is_weekend          BOOL    NOT NULL,
  is_public_holiday   BOOL    NOT NULL   -- nationwide German public holidays
);

CREATE TABLE ecom_v1.categories (
  category_id         INT64   NOT NULL,  -- PK
  category_name       STRING  NOT NULL,
  parent_category_id  INT64               -- NULL = top-level category, else self-FK
);

CREATE TABLE ecom_v1.products (
  product_id          INT64   NOT NULL,  -- PK
  sku                 STRING  NOT NULL,
  product_name        STRING  NOT NULL,
  category_id         INT64   NOT NULL,  -- FK -> categories
  list_price          NUMERIC NOT NULL,  -- catalog price, NOT the price actually charged
  purchase_price      NUMERIC NOT NULL,  -- cost of goods, basis for margin
  launched_on         DATE    NOT NULL,
  discontinued_on     DATE                -- NULL = still in assortment
);

CREATE TABLE ecom_v1.customers (
  customer_id         INT64   NOT NULL,  -- PK
  customer_number     STRING  NOT NULL,
  signup_date         DATE    NOT NULL,
  country_code        STRING  NOT NULL,  -- 'DE', 'AT', 'CH'
  postal_code         STRING,
  segment             STRING  NOT NULL,  -- 'private', 'business'
  newsletter_optin    BOOL    NOT NULL,
  deleted_at          TIMESTAMP           -- NULL = account exists (GDPR erasure)
);

CREATE TABLE ecom_v1.channels (
  channel_id          INT64   NOT NULL,  -- PK
  channel_name        STRING  NOT NULL,  -- 'webshop', 'app', 'marketplace_a', 'phone'
  channel_type        STRING  NOT NULL,  -- 'direct', 'partner'
  commission_rate     NUMERIC NOT NULL   -- 0.00 for direct channels
);

CREATE TABLE ecom_v1.orders (
  order_id            INT64   NOT NULL,  -- PK
  customer_id         INT64   NOT NULL,  -- FK -> customers
  channel_id          INT64   NOT NULL,  -- FK -> channels
  order_date          DATE    NOT NULL,  -- date the order was placed
  ship_date           DATE,               -- NULL = not shipped yet
  order_status        STRING  NOT NULL,  -- 'placed','shipped','delivered','cancelled'
  shipping_cost       NUMERIC NOT NULL,
  payment_method      STRING  NOT NULL   -- 'card','invoice','paypal','direct_debit'
);

CREATE TABLE ecom_v1.order_items (
  order_item_id       INT64   NOT NULL,  -- PK
  order_id            INT64   NOT NULL,  -- FK -> orders
  product_id          INT64   NOT NULL,  -- FK -> products
  quantity            INT64   NOT NULL,
  unit_price_gross    NUMERIC NOT NULL,  -- price actually charged per unit, incl. VAT
  discount_amount     NUMERIC NOT NULL,  -- absolute discount on the line item, >= 0
  tax_rate            NUMERIC NOT NULL   -- 0.19 or 0.07
);

CREATE TABLE ecom_v1.returns (
  return_id           INT64   NOT NULL,  -- PK
  order_item_id       INT64   NOT NULL,  -- FK -> order_items
  return_date         DATE    NOT NULL,
  quantity_returned   INT64   NOT NULL,  -- may be < quantity (partial return)
  reason_code         STRING  NOT NULL,  -- 'defect','wrong_size','not_as_described','no_reason'
  refund_amount       NUMERIC NOT NULL
);
```

Longest natural join chain (4 tables, no construction needed):
`returns → order_items → orders → customers`, or via `→ products → categories`.

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

## Column traps

| Trap | Columns | Rule |
|---|---|---|
| Similar date columns | `order_date` vs. `ship_date` vs. `return_date`, also `signup_date`, `launched_on` | Default to `order_date` unless the question explicitly asks about shipping/fulfillment/returns |
| Three price columns | `list_price` vs. `unit_price_gross` vs. `purchase_price` | `list_price` is catalog price and is almost never the right column for revenue |
| Partial returns | `returns.quantity_returned <= order_items.quantity` | A return is not automatically a full cancellation of the line item |
| Self-reference | `categories.parent_category_id` | "Revenue by main category" needs a self-join or recursive CTE, not a flat `GROUP BY category_id` |
| Meaningful NULLs | `ship_date`, `discontinued_on`, `deleted_at` | NULL means something specific, not "unknown" |
| Cancellation semantics | `order_status = 'cancelled'` | Exclude cancelled orders only for revenue, contribution-margin, customer-activity, repeat-purchasing, or quantity-sold metrics. For a plain count of placed orders, include them unless the question explicitly says otherwise. |

---

## Output conventions

- Return category, product, customer, and channel **names**, not internal
  IDs, unless the question explicitly asks for an ID.
- Return only the columns the question asks for. Do not add explanatory or
  derived columns beyond what is requested.
- Write plain BigQuery Standard SQL. No comments, no markdown fences, no
  trailing semicolon requirement — a single query per answer.
