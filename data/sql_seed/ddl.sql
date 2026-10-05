CREATE TABLE ecom_v1.dim_date (
  date_key            DATE    NOT NULL,
  year                INT64   NOT NULL,
  quarter             INT64   NOT NULL,
  month               INT64   NOT NULL,
  day_of_month        INT64   NOT NULL,
  day_of_week         INT64   NOT NULL,   -- 1 = Monday
  iso_week            INT64   NOT NULL,
  is_weekend          BOOL    NOT NULL,
  is_public_holiday   BOOL    NOT NULL    -- nationwide German public holidays
);

CREATE TABLE ecom_v1.categories (
  category_id         INT64   NOT NULL,
  category_name       STRING  NOT NULL,
  parent_category_id  INT64                -- NULL = top-level category, else self-reference
);

CREATE TABLE ecom_v1.products (
  product_id          INT64   NOT NULL,
  sku                 STRING  NOT NULL,
  product_name        STRING  NOT NULL,
  category_id         INT64   NOT NULL,
  list_price          NUMERIC NOT NULL,    -- catalog price, NOT the price actually charged
  purchase_price      NUMERIC NOT NULL,    -- cost of goods, basis for margin
  launched_on         DATE    NOT NULL,
  discontinued_on     DATE                 -- NULL = still in assortment
);

CREATE TABLE ecom_v1.customers (
  customer_id         INT64   NOT NULL,
  customer_number     STRING  NOT NULL,
  signup_date         DATE    NOT NULL,
  country_code        STRING  NOT NULL,    -- 'DE', 'AT', 'CH'
  postal_code         STRING,
  segment             STRING  NOT NULL,    -- 'private', 'business'
  newsletter_optin    BOOL    NOT NULL,
  deleted_at          TIMESTAMP            -- NULL = account exists (GDPR erasure)
);

CREATE TABLE ecom_v1.channels (
  channel_id          INT64   NOT NULL,
  channel_name        STRING  NOT NULL,    -- 'webshop', 'app', 'marketplace_a', 'phone'
  channel_type        STRING  NOT NULL,    -- 'direct', 'partner'
  commission_rate     NUMERIC NOT NULL     -- 0.00 for direct channels
);

CREATE TABLE ecom_v1.orders (
  order_id            INT64   NOT NULL,
  customer_id         INT64   NOT NULL,
  channel_id          INT64   NOT NULL,
  order_date          DATE    NOT NULL,    -- date the order was placed
  ship_date           DATE,                -- NULL = not shipped yet
  order_status        STRING  NOT NULL,    -- 'placed','shipped','delivered','cancelled'
  shipping_cost       NUMERIC NOT NULL,
  payment_method      STRING  NOT NULL     -- 'card','invoice','paypal','direct_debit'
);

CREATE TABLE ecom_v1.order_items (
  order_item_id       INT64   NOT NULL,
  order_id            INT64   NOT NULL,
  product_id          INT64   NOT NULL,
  quantity            INT64   NOT NULL,
  unit_price_gross    NUMERIC NOT NULL,    -- price actually charged per unit, incl. VAT
  discount_amount     NUMERIC NOT NULL,    -- absolute discount on the line item, >= 0
  tax_rate            NUMERIC NOT NULL     -- 0.19 or 0.07
);

CREATE TABLE ecom_v1.returns (
  return_id           INT64   NOT NULL,
  order_item_id       INT64   NOT NULL,
  return_date         DATE    NOT NULL,
  quantity_returned   INT64   NOT NULL,    -- may be < quantity (partial return)
  reason_code         STRING  NOT NULL,    -- 'defect','wrong_size','not_as_described','no_reason'
  refund_amount       NUMERIC NOT NULL
);
