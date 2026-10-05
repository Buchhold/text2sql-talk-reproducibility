DATASET=example-project:ecom_v1

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.dim_date ./seed/dim_date.csv \
  date_key:DATE,year:INT64,quarter:INT64,month:INT64,day_of_month:INT64,day_of_week:INT64,iso_week:INT64,is_weekend:BOOL,is_public_holiday:BOOL

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.categories ./seed/categories.csv \
  category_id:INT64,category_name:STRING,parent_category_id:INT64

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.products ./seed/products.csv \
  product_id:INT64,sku:STRING,product_name:STRING,category_id:INT64,list_price:NUMERIC,purchase_price:NUMERIC,launched_on:DATE,discontinued_on:DATE

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.customers ./seed/customers.csv \
  customer_id:INT64,customer_number:STRING,signup_date:DATE,country_code:STRING,postal_code:STRING,segment:STRING,newsletter_optin:BOOL,deleted_at:TIMESTAMP

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.channels ./seed/channels.csv \
  channel_id:INT64,channel_name:STRING,channel_type:STRING,commission_rate:NUMERIC

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.orders ./seed/orders.csv \
  order_id:INT64,customer_id:INT64,channel_id:INT64,order_date:DATE,ship_date:DATE,order_status:STRING,shipping_cost:NUMERIC,payment_method:STRING

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.order_items ./seed/order_items.csv \
  order_item_id:INT64,order_id:INT64,product_id:INT64,quantity:INT64,unit_price_gross:NUMERIC,discount_amount:NUMERIC,tax_rate:NUMERIC

bq load --source_format=CSV --skip_leading_rows=1 --replace \
  $DATASET.returns ./seed/returns.csv \
  return_id:INT64,order_item_id:INT64,return_date:DATE,quantity_returned:INT64,reason_code:STRING,refund_amount:NUMERIC
