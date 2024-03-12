# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")
dbutils.widgets.text("base_dir", "")

# COMMAND ----------

import os
import pandas as pd
import pyspark.sql.functions as f

datamart_dir = dbutils.widgets.get("base_dir") + "/datamart"

sales = spark.read.parquet(os.path.join(datamart_dir, "transaction.parquet"))
vip = spark.read.parquet(os.path.join(datamart_dir, "demographic.parquet"))
first_purchase = spark.read.parquet(os.path.join(datamart_dir, "first_purchase.parquet"))
# item_attr_tagging = spark.read.parquet(os.path.join(datamart_dir, "item_attr_tagging.parquet"))
item_attr_tagging = spark.read.parquet("/mnt/dev/customer_segmentation/imx/club_monaco/datamart/item_attr_tagging.parquet")

# COMMAND ----------

feature_dir = dbutils.widgets.get("base_dir") + "/features"
os.makedirs(feature_dir, exist_ok=True)

# COMMAND ----------

def save_feature_df(df, filename):
    df.write.parquet(os.path.join(feature_dir, f"{filename}.parquet"), mode="overwrite")

# COMMAND ----------

sales.createOrReplaceTempView("sales0")
vip.createOrReplaceTempView("vip")
first_purchase.createOrReplaceTempView("first_purchase")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW sales AS
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   item_subcat_desc AS item_subcat_desc_cleaned,
# MAGIC   maincat_desc AS maincat_desc_cleaned
# MAGIC FROM sales0
# MAGIC WHERE
# MAGIC   isnull(vip_main_no) = 0 AND vip_main_no != ""
# MAGIC   AND isnull(prod_brand) = 0 AND prod_brand NOT IN ("JBZZZ", "ZZ")
# MAGIC   AND isnull(item_subcat_desc) = 0 AND item_subcat_desc NOT IN ("ZZZ", "Dummy", "dummy", "GIFT", "gift")
# MAGIC   AND isnull(maincat_desc) = 0 AND maincat_desc NOT IN ("ZZZ", "Dummy", "dummy", "GIFT")

# COMMAND ----------

sales = spark.table("sales")

# COMMAND ----------

# MAGIC %md
# MAGIC product features

# COMMAND ----------

def features_in_list_by_vip(feature, table=sales):
    grouped_df = table.groupBy("vip_main_no").agg(f.collect_list(feature).alias(feature))
    return grouped_df


# COMMAND ----------

def count_encoding(feature, table=sales, prefix="", postfix="_qty"):
    table = table.filter((f.col(feature).isNotNull()) & (f.col(feature) != ""))
    df = (
        table.groupBy("vip_main_no")
        .pivot(feature)
        .agg(f.sum("sold_qty"))
        .fillna(0)
    )
    renamed_columns = ["vip_main_no"] + [prefix + col + postfix for col in df.columns if col != "vip_main_no"]
    df = df.toDF(*renamed_columns)
    return df


# COMMAND ----------

subcat = count_encoding("item_subcat_desc_cleaned", prefix="subcat_")
save_feature_df(subcat, "subcat")

# COMMAND ----------

maincat = count_encoding("maincat_desc_cleaned", prefix="maincat_")
save_feature_df(maincat, "maincat")

# COMMAND ----------

prod_brand = count_encoding("prod_brand", prefix="brand_")
save_feature_df(prod_brand, "brand")

# COMMAND ----------

# tags
temp = sales.join(item_attr_tagging.select("item_desc", "tags"), on="item_desc", how="left")
exploded_df = temp.select("vip_main_no", f.explode("tags").alias("tags"), "sold_qty")
tagging = count_encoding("tags", table=exploded_df, prefix="tag_")
save_feature_df(tagging, "tagging")

# COMMAND ----------

# MAGIC %md
# MAGIC demographic

# COMMAND ----------

demographic = spark.sql("""with tenure as (
  Select
    distinct
    vip_main_no,
    first_pur_cm,
    round(
      datediff(
        TO_DATE(CONCAT(YEAR(getArgument("start_date")),'1231'), "yyyyMMdd"),
        first_pur_cm
      ) / 365,
      0
    ) as tenure
  from
    first_purchase
)
select
  vip_main_no,
  min(
    case
      when customer_sex = "C"
      OR isnull(customer_sex) = 1
      OR customer_sex = "" then "C"
      else customer_sex
    end
  ) as customer_sex,
  min(
    case
      when cust_nat_cat = "Hong Kong" then "Hong Kong"
      when cust_nat_cat = "Mainland China" then "Mainland China"
      when cust_nat_cat = "Macau" then "Macau"
      else "Others"
    end
  ) as cust_nat_cat,
  case
    when tenure <= 1 then '0-1'
    when tenure > 1
    and tenure <= 3 then '1-3'
    when tenure > 3
    and tenure <= 7 then '3-7'
    else '8+'
  end as tenure,
  max(case 
    when customer_age_group = '01' then '< 25'
    when customer_age_group = '02' then '26 - 30'
    when customer_age_group = '03' then '31 - 35'
    when customer_age_group = '04' then '36 - 40'
    when customer_age_group = '05' then '41 - 50'
    when customer_age_group = '06' then '> 51'
    when customer_age_group = '07' then null
  else null end) as age
from
  sales
  left join tenure using (vip_main_no)
group by
  1,
  4
""")
