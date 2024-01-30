# Databricks notebook source
# MAGIC %py
# MAGIC dbutils.widgets.removeAll()
# MAGIC dbutils.widgets.text("start_date", "2023-01-01")
# MAGIC dbutils.widgets.text("end_date", "2023-12-31")

# COMMAND ----------

import os
import pyspark.sql.functions as f

base_dir = "/mnt/dev/customer_segmentation/imx/club_monaco/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
vip = spark.read.parquet(os.path.join(base_dir, "demographic.parquet"))
first_purchase = spark.read.parquet(os.path.join(base_dir, "first_purchase.parquet"))
sales.createOrReplaceTempView("sales")
vip.createOrReplaceTempView("vip")
first_purchase.createOrReplaceTempView("first_purchase")

# COMMAND ----------

# clustering result
persona = spark.read.parquet(
    "/mnt/dev/customer_segmentation/imx/club_monaco/model/clustering_result_kmeans_iter3.parquet")
persona.createOrReplaceTempView("persona0")


# COMMAND ----------

sub_persona = spark.read.parquet(
    "/mnt/dev/customer_segmentation/imx/club_monaco/model/clustering_result_sub_kmeans_iter3.parquet")
sub_persona.createOrReplaceTempView("sub_persona0")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW persona1 as
# MAGIC SELECT a.vip_main_no,
# MAGIC   CASE WHEN a.persona = 2 AND b.persona = 0 THEN 2
# MAGIC             WHEN a.persona = 2 AND b.persona = 1 THEN 5
# MAGIC             WHEN a.persona = 2 AND b.persona = 2 THEN 6
# MAGIC             ELSE a.persona
# MAGIC        END AS persona
# MAGIC FROM persona0 a
# MAGIC LEFT JOIN sub_persona0 b ON a.vip_main_no = b.vip_main_no;

# COMMAND ----------

# MAGIC %md
# MAGIC persona
# MAGIC
# MAGIC 0&1&2: Average Fashion Connoisseur
# MAGIC
# MAGIC 3: Prime Connoisseur
# MAGIC
# MAGIC 4: Menswear Specialist
# MAGIC
# MAGIC 5: Outerwear Fashionista
# MAGIC
# MAGIC 6: Bottoms and Dresses Diva

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW persona AS
# MAGIC SELECT
# MAGIC   vip_main_no,
# MAGIC   CASE WHEN persona = 0 THEN "Average Fashion Connoisseur"
# MAGIC   WHEN persona = 1 THEN "Average Fashion Connoisseur"
# MAGIC   WHEN persona = 2 THEN "Average Fashion Connoisseur"
# MAGIC   WHEN persona = 3 THEN "Prime Connoisseur"
# MAGIC   WHEN persona = 4 THEN "Menswear Specialist"
# MAGIC   WHEN persona = 5 THEN "Outerwear Fashionista"
# MAGIC   WHEN persona = 6 THEN "Bottoms and Dresses Diva" END AS persona
# MAGIC FROM persona1

# COMMAND ----------

cluster_order = ["Average Fashion Connoisseur", "Prime Connoisseur", "Menswear Specialist", "Outerwear Fashionista", "Bottoms and Dresses Diva"]

# COMMAND ----------


def sum_pivot_table(table, group_by_col, agg_col, show_inactive=True):
    df = table.groupBy("customer_tag", group_by_col).agg(f.sum(agg_col))
    pivot_table = (
        df.groupBy(group_by_col).pivot("customer_tag").agg(f.sum(f"sum({agg_col})"))
    )
    display(pivot_table.select(group_by_col, *cluster_order))
    return pivot_table


def count_pivot_table(table, group_by_col, agg_col, percentage=False, show_inactive=True):
    df = table.groupBy("customer_tag", group_by_col).agg(f.countDistinct(agg_col).alias("count"))
    pivot_table = (
        df.groupBy(group_by_col)
        .pivot("customer_tag")
        .agg(f.sum(f"count"))
    )
    display(pivot_table.select(group_by_col, *cluster_order))
    return pivot_table


# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW sales_cleaned AS
# MAGIC WITH cte1 AS (
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   CASE WHEN maincat_desc = "gift" THEN "GIFT"
# MAGIC   WHEN maincat_desc = "ZZ" OR maincat_desc = "Dummy" THEN "Unknown" ELSE maincat_desc END AS maincat_desc_cleaned,
# MAGIC   CASE WHEN item_subcat_desc = "ZZZ" THEN "Unknown" 
# MAGIC   WHEN item_subcat_desc = "dummy" THEN "Unknown" 
# MAGIC   WHEN item_subcat_desc = "Dummy" THEN "Unknown" 
# MAGIC   WHEN item_subcat_desc = "gift" THEN "GIFT" ELSE item_subcat_desc END AS item_subcat_desc_cleaned
# MAGIC from sales
# MAGIC ),
# MAGIC Cte2 AS (
# MAGIC   select 
# MAGIC     *,
# MAGIC     concat(maincat_desc_cleaned, " - ", item_subcat_desc_cleaned) as maincat_and_subcat 
# MAGIC   from cte1
# MAGIC )
# MAGIC SELECT * FROM Cte2
# MAGIC WHERE prod_brand = "CM"
# MAGIC AND maincat_desc_cleaned not in ("Unknown", "GIFT")

# COMMAND ----------

final_sales_table = spark.sql(
    """
    select *, 1 as dummy, persona as customer_tag from sales_cleaned
    inner join persona using (vip_main_no)
    """
)
final_sales_table.createOrReplaceTempView("final_sales_table")

# COMMAND ----------

# MAGIC %md
# MAGIC demographic

# COMMAND ----------

# count of customer
count_pivot_table(final_sales_table, group_by_col="dummy", agg_col="vip_main_no")

# COMMAND ----------

# gender
df = spark.sql(
    """
    with tem as (select distinct vip_main_no, case when customer_sex = "C" OR isnull(customer_sex) = 1 then "C"
        else customer_sex end as customer_sex_new,
        customer_tag
        from final_sales_table)
        select distinct vip_main_no, min(customer_sex_new) as customer_sex_new, customer_tag from tem group by vip_main_no, customer_tag

    """
)
count_pivot_table(df, group_by_col="customer_sex_new", agg_col="vip_main_no")

# COMMAND ----------

# tenure

# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temporary view tenure as
# MAGIC Select
# MAGIC   vip_main_no,
# MAGIC   first_pur_cm,
# MAGIC   round(
# MAGIC     datediff(
# MAGIC       TO_DATE("20231231", "yyyyMMdd"),
# MAGIC       first_pur_cm
# MAGIC     ) / 365,
# MAGIC     0
# MAGIC   ) as tenure
# MAGIC from
# MAGIC   first_purchase
# MAGIC   inner join sales_cleaned using (vip_main_no)

# COMMAND ----------

df = spark.sql("""select
  distinct vip_main_no,
  max(case
    when tenure <= 1 then '0-1'
    when tenure > 1
    and tenure <= 3 then '1-3'
    when tenure > 3
    and tenure <= 7 then '3-7'
    else '8+'
  end) as tenure,
  persona as customer_tag,
  max(tenure)
from
  tenure
  inner join persona using (vip_main_no)
group by 1, 3
""")

count_pivot_table(df, group_by_col="tenure", agg_col="vip_main_no")

# COMMAND ----------

# nationality

df = spark.sql(
    """
    with tem as (select *,
    case when cust_nat_cat = "Hong Kong" then "Hong Kong" 
    when cust_nat_cat = "Mainland China" then "Mainland China" 
    when cust_nat_cat = "Macau" then "Macau" 
    else "Others" end as cust_nat_cat_new
    from final_sales_table)
    select distinct vip_main_no, min(cust_nat_cat_new) cust_nat_cat_new, customer_tag from tem group by vip_main_no, customer_tag
    """
)
count_pivot_table(df, group_by_col="cust_nat_cat_new", agg_col="vip_main_no")

# COMMAND ----------

# age group
df = spark.sql(
    """
    select distinct vip_main_no, max(customer_age_group) customer_age_group, customer_tag from final_sales_table group by vip_main_no, customer_tag
    """
)

table = count_pivot_table(df, group_by_col="customer_age_group", agg_col="vip_main_no").createOrReplaceTempView(
    "age_gp")

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   distinct
# MAGIC   case 
# MAGIC     when customer_age_group = '01' then '< 25'
# MAGIC     when customer_age_group = '02' then '26 - 30'
# MAGIC     when customer_age_group = '03' then '31 - 35'
# MAGIC     when customer_age_group = '04' then '36 - 40'
# MAGIC     when customer_age_group = '05' then '41 - 50'
# MAGIC     when customer_age_group = '06' then '> 51'
# MAGIC     when customer_age_group = '07' then null
# MAGIC   else null end as age,
# MAGIC   sum(`Average Fashion Connoisseur`),
# MAGIC   sum(`Prime Connoisseur`),
# MAGIC   sum(`Menswear Specialist`),
# MAGIC   sum(`Outerwear Fashionista`),
# MAGIC   sum(`Bottoms and Dresses Diva`)
# MAGIC
# MAGIC from age_gp
# MAGIC group by age

# COMMAND ----------

# MAGIC %md
# MAGIC transactional

# COMMAND ----------

# amt
df = spark.sql(
    """
    select * from final_sales_table
    where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
    """
)
sum_pivot_table(df, group_by_col="dummy", agg_col="net_amt_hkd", show_inactive=False)

# COMMAND ----------

# qty
sum_pivot_table(df, group_by_col="dummy", agg_col="sold_qty", show_inactive=False)

# COMMAND ----------

# # of order
count_pivot_table(df, group_by_col="dummy", agg_col="invoice_no", show_inactive=False)

# COMMAND ----------

# # of visit
visit = spark.sql(
    """with visit as (
select
  distinct vip_main_no,
  order_date,
  shop_code,
  customer_tag
from final_sales_table
 where order_date >= getArgument("start_date") and order_date <= getArgument("end_date") 
)
select 
  vip_main_no,
  order_date,
  shop_code,
  customer_tag,
  count(distinct vip_main_no,
  order_date,
  shop_code) as visit,
  1 as dummy
from visit
group by
  vip_main_no,
  order_date,
  shop_code,
  customer_tag
""")

sum_pivot_table(visit, group_by_col="dummy", agg_col="visit", show_inactive=False)


# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temporary view new_joiner as
# MAGIC Select
# MAGIC   f.vip_main_no,
# MAGIC   first_pur_cm,
# MAGIC   CASE
# MAGIC     WHEN first_pur_cm >= TO_DATE("20230101", "yyyyMMdd") THEN 1
# MAGIC     ELSE 0
# MAGIC   END AS new_joiner_flag
# MAGIC from
# MAGIC   first_purchase f

# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temporary view visit as with visit0 as (
# MAGIC   select
# MAGIC     distinct vip_main_no,
# MAGIC     order_date,
# MAGIC     shop_code,
# MAGIC     customer_tag
# MAGIC   from
# MAGIC     final_sales_table
# MAGIC   where
# MAGIC     order_date >= getArgument("start_date")
# MAGIC     and order_date <= getArgument("end_date")
# MAGIC )
# MAGIC select
# MAGIC   vip_main_no,
# MAGIC   customer_tag,
# MAGIC   count(distinct vip_main_no, order_date, shop_code) as visit
# MAGIC from
# MAGIC   visit0
# MAGIC group by
# MAGIC   vip_main_no,
# MAGIC   customer_tag

# COMMAND ----------

# new joiners repeat purchase
df = spark.sql("""
               select * from visit
               inner join (
                   select * from new_joiner
                   where new_joiner_flag = 1
                   ) using (vip_main_no)
               
               """)
df = df.groupBy("customer_tag", "visit").agg(f.countDistinct("vip_main_no").alias("count"))
pivot_table = (
    df.groupBy("visit")
    .pivot("customer_tag")
    .agg(f.sum(f"count"))
)
display(pivot_table)

# COMMAND ----------

# of new joiners
df = spark.sql("""select
  distinct vip_main_no,
  new_joiner_flag,
  persona as customer_tag
from
  new_joiner
  inner join persona using (vip_main_no)
""")

df = df.groupBy("customer_tag", "new_joiner_flag").agg(f.countDistinct("vip_main_no").alias("count"))
pivot_table = (
    df.groupBy("new_joiner_flag")
    .pivot("customer_tag")
    .agg(f.sum(f"count"))
)
display(pivot_table)

# COMMAND ----------

df = spark.sql("""select
  distinct vip_main_no,
  new_joiner_flag,
  visit,
  persona as customer_tag
from
  new_joiner
  inner join visit using (vip_main_no)
  inner join persona using (vip_main_no)
where persona = "Average Fashion Connoisseur"
""")

df = df.groupBy("customer_tag", "visit").agg(f.countDistinct("vip_main_no").alias("count"))
pivot_table = (
    df.groupBy("visit")
    .pivot("customer_tag")
    .agg(f.sum(f"count"))
)
display(pivot_table)

# COMMAND ----------

# MAGIC %sql
# MAGIC create
# MAGIC or replace temp view df as
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   final_sales_table
# MAGIC   inner join (
# MAGIC     select
# MAGIC       vip_main_no
# MAGIC     from
# MAGIC       new_joiner
# MAGIC       inner join visit using (vip_main_no)
# MAGIC     where
# MAGIC       new_joiner_flag = 1
# MAGIC       and visit > 1
# MAGIC   ) using (vip_main_no)

# COMMAND ----------

df = spark.table("df")

# COMMAND ----------

count_pivot_table(df, group_by_col="dummy", agg_col="vip_main_no")

# COMMAND ----------

sum_pivot_table(df, group_by_col="dummy", agg_col="net_amt_hkd")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- MEMBER PENETRATION BY STORE
# MAGIC -- cust count by store and segment
# MAGIC select * from
# MAGIC (select 
# MAGIC     distinct shop_desc,
# MAGIC     customer_tag, 
# MAGIC     count(distinct vip_main_no) as vip_count
# MAGIC from final_sales_table
# MAGIC where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
# MAGIC group by 
# MAGIC     customer_tag,
# MAGIC     shop_desc
# MAGIC )
# MAGIC PIVOT (
# MAGIC   SUM(vip_count)
# MAGIC   FOR customer_tag IN ('Average Fashion Connoisseur',
# MAGIC  'Prime Connoisseur',
# MAGIC  'Menswear Specialist',
# MAGIC  'Outerwear Fashionista',
# MAGIC  'Bottoms and Dresses Diva')
# MAGIC ) 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from
# MAGIC (select 
# MAGIC     distinct shop_desc,
# MAGIC     customer_tag, 
# MAGIC     sum(net_amt_hkd) as amt
# MAGIC from final_sales_table
# MAGIC where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
# MAGIC group by 
# MAGIC     customer_tag,
# MAGIC     shop_desc
# MAGIC )
# MAGIC PIVOT (
# MAGIC   SUM(amt)
# MAGIC   FOR customer_tag IN ('Average Fashion Connoisseur',
# MAGIC  'Prime Connoisseur',
# MAGIC  'Menswear Specialist',
# MAGIC  'Outerwear Fashionista',
# MAGIC  'Bottoms and Dresses Diva')
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Member penetration by month
# MAGIC -- cust count by yearmon and segment
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   (
# MAGIC     select
# MAGIC       distinct yyyymm,
# MAGIC       customer_tag,
# MAGIC       count(distinct vip_main_no) as vip_count
# MAGIC     from
# MAGIC       (
# MAGIC         select
# MAGIC           *,
# MAGIC           CONCAT(
# MAGIC             year(order_date),
# MAGIC             LPAD(month(order_date), 2, '0')
# MAGIC           ) as yyyymm
# MAGIC         from
# MAGIC           final_sales_table
# MAGIC       )
# MAGIC     group by
# MAGIC       customer_tag,
# MAGIC       yyyymm
# MAGIC   ) PIVOT (
# MAGIC     SUM(vip_count) FOR customer_tag IN ('Average Fashion Connoisseur',
# MAGIC  'Prime Connoisseur',
# MAGIC  'Menswear Specialist',
# MAGIC  'Outerwear Fashionista',
# MAGIC  'Bottoms and Dresses Diva')
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   (
# MAGIC     select
# MAGIC       distinct yyyymm,
# MAGIC       customer_tag,
# MAGIC       sum(net_amt_hkd) as amt
# MAGIC     from
# MAGIC       (
# MAGIC         select
# MAGIC           *,
# MAGIC           CONCAT(
# MAGIC             year(order_date),
# MAGIC             LPAD(month(order_date), 2, '0')
# MAGIC           ) as yyyymm
# MAGIC         from
# MAGIC           final_sales_table
# MAGIC       )
# MAGIC     group by
# MAGIC       customer_tag,
# MAGIC       yyyymm
# MAGIC   ) PIVOT (
# MAGIC     SUM(amt) FOR customer_tag IN ('Average Fashion Connoisseur',
# MAGIC  'Prime Connoisseur',
# MAGIC  'Menswear Specialist',
# MAGIC  'Outerwear Fashionista',
# MAGIC  'Bottoms and Dresses Diva')
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC features based on class, subclass
# MAGIC

# COMMAND ----------

# - share of wallet 
# - AVERAGE ITEM VALUE
# - MEMBER PENETRATION
# - $ SPEND PER MEMBER

# COMMAND ----------

def pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="net_amt_hkd", mode="sum",
                       table="final_sales_table"):
    df = spark.sql(
        f"""
        select * from
            (select 
                distinct 
                case when isnull({group_by}) = 1 or {group_by} = "N/A" then "Unknown" else {group_by} end as {group_by},
                customer_tag, 
                {mode}({agg_col}) as overall_amount
            from {table}
            where order_date >= getArgument("start_date") and order_date <= getArgument("end_date")
            group by 
                customer_tag,
                {group_by}
            )
            PIVOT (
            SUM(overall_amount)
            FOR customer_tag IN ('Average Fashion Connoisseur',
                'Prime Connoisseur',
                'Menswear Specialist',
                'Outerwear Fashionista',
                'Bottoms and Dresses Diva')
            )
        """
    )
    display(df)
    return df


# COMMAND ----------

# by maincat and subcat

# COMMAND ----------

# 1. amt table by subclass and segment
pivot_table_by_cat(group_by="maincat_and_subcat", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------

# 2. qty table by subclass and segment
pivot_table_by_cat(group_by="maincat_and_subcat", agg_col="sold_qty", mode="sum")

# COMMAND ----------

# 3. number of member purchase by subclass and segment
pivot_table_by_cat(group_by="maincat_and_subcat", agg_col="distinct vip_main_no", mode="count")

# COMMAND ----------

# by subclass

# COMMAND ----------

# 1. amt table by subclass and segment
pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------

# 2. qty table by subclass and segment
pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="sold_qty", mode="sum")

# COMMAND ----------

# 3. number of member purchase by subclass and segment
pivot_table_by_cat(group_by="item_subcat_desc_cleaned", agg_col="distinct vip_main_no", mode="count")

# COMMAND ----------

# by maincat_desc

# COMMAND ----------

# 1. amt table by maincat_desc and segment
pivot_table_by_cat(group_by="maincat_desc_cleaned", agg_col="net_amt_hkd", mode="sum")

# COMMAND ----------

# 2. qty table by maincat_desc and segment
pivot_table_by_cat(group_by="maincat_desc_cleaned", agg_col="sold_qty", mode="sum")

# COMMAND ----------

# 3. number of member purchase by maincat_desc and segment
pivot_table_by_cat(group_by="maincat_desc_cleaned", agg_col="distinct vip_main_no", mode="count")

# COMMAND ----------

# MAGIC %md
# MAGIC item tagging

# COMMAND ----------

item_attr = spark.read.parquet(os.path.join(base_dir, "item_attr_tagging.parquet")).toPandas()
item_attr_exploded = item_attr.explode("tags")
item_attr_exploded_spark = spark.createDataFrame(item_attr_exploded)
final_sales_table_with_tags = final_sales_table.join(item_attr_exploded_spark.select("item_desc", "tags"),
                                                     on="item_desc", how="left")

# COMMAND ----------

final_sales_table_with_tags.createOrReplaceTempView("final_sales_table_with_tags")

# COMMAND ----------

# tagging parquet for cross-class behaviour
item_attr_1 = spark.read.parquet(os.path.join(base_dir, "item_attr_tagging1.parquet")).toPandas()
item_attr_exploded_1 = item_attr_1.explode("tags")
item_attr_exploded_spark_1 = spark.createDataFrame(item_attr_exploded_1)
final_sales_table_with_tags_1 = final_sales_table.join(item_attr_exploded_spark_1.select("item_desc", "tags"),
                                                     on="item_desc", how="left")

# COMMAND ----------

final_sales_table_with_tags_1.createOrReplaceTempView("final_sales_table_with_tags_1")

# COMMAND ----------

# 1. amt table by tag and segment
pivot_table_by_cat(group_by="tags", agg_col="net_amt_hkd", mode="sum", table="final_sales_table_with_tags")

# COMMAND ----------

# 1. amt table by tag and segment for cross-class behaviour
pivot_table_by_cat(group_by="tags", agg_col="net_amt_hkd", mode="sum", table="final_sales_table_with_tags_1")

# COMMAND ----------

# 2. qty table by tag and segment
pivot_table_by_cat(group_by="tags", agg_col="sold_qty", mode="sum", table="final_sales_table_with_tags")

# COMMAND ----------

# 2. qty table by tag and segment for cross-class behaviour
pivot_table_by_cat(group_by="tags", agg_col="sold_qty", mode="sum", table="final_sales_table_with_tags_1")

# COMMAND ----------

# 3. number of member purchase by tag and segment
pivot_table_by_cat(group_by="tags", agg_col="distinct vip_main_no", mode="count", table="final_sales_table_with_tags")
