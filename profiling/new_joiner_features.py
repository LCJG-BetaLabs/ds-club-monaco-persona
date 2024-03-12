# Databricks notebook source
# MAGIC %sql 
# MAGIC create or replace temp view nj_sales as
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     vip_main_no, 
# MAGIC     new_joiner_flag,
# MAGIC     sales_staff_flag,
# MAGIC     maincat_desc_cleaned,
# MAGIC     item_subcat_desc_cleaned,
# MAGIC     maincat_and_subcat,
# MAGIC     order_date,
# MAGIC     customer_tag,
# MAGIC     item_desc 
# MAGIC   FROM 
# MAGIC     new_joiner 
# MAGIC   INNER JOIN 
# MAGIC     final_sales_table 
# MAGIC   USING (vip_main_no)
# MAGIC ) subquery
# MAGIC WHERE 
# MAGIC   subquery.new_joiner_flag <> 0
# MAGIC   AND subquery.sales_staff_flag = 0;

# COMMAND ----------

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

cluster_order = ["Prime Connoisseur", "Menswear Specialist", "Outerwear Fashionista", "Bottoms and Dresses Diva", "Average Fashion Connoisseur"]

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
# MAGIC AND sales_staff_flag = 0

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

njsales = spark.sql(
    """
    select * from final_sales_table
    inner join new_joiner using (vip_main_no)
    where new_joiner_flag = 1
    """
)
njsales.createOrReplaceTempView("njsales")

# COMMAND ----------

# count of customer
count_pivot_table(njsales, group_by_col="dummy", agg_col="vip_main_no")

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

# DBTITLE 1,new joiner sales table
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW nj_sales AS
# MAGIC SELECT *
# MAGIC FROM new_joiner
# MAGIC INNER JOIN sales_cleaned
# MAGIC USING (vip_main_no)
# MAGIC WHERE new_joiner_flag = 1
# MAGIC AND sales_staff_flag = 0;

# COMMAND ----------

#new joiner gender
# gender
df = spark.sql(
    """
    with tem as (select distinct vip_main_no, case when customer_sex = "C" OR isnull(customer_sex) = 1 then "C"
        else customer_sex end as customer_sex_new,
        customer_tag
        from nj_sales)
        select distinct vip_main_no, min(customer_sex_new) as customer_sex_new, customer_tag from tem group by vip_main_no, customer_tag

    """
)
count_pivot_table(df, group_by_col="customer_sex_new", agg_col="vip_main_no")

# COMMAND ----------

# nj nationality

df = spark.sql(
    """
    with tem as (select *,
    case when cust_nat_cat = "Hong Kong" then "Hong Kong" 
    when cust_nat_cat = "Mainland China" then "Mainland China" 
    when cust_nat_cat = "Macau" then "Macau" 
    else "Others" end as cust_nat_cat_new
    from nj_sales)
    select distinct vip_main_no, min(cust_nat_cat_new) cust_nat_cat_new, customer_tag from tem group by vip_main_no, customer_tag
    """
)
count_pivot_table(df, group_by_col="cust_nat_cat_new", agg_col="vip_main_no")

# COMMAND ----------

# nj age group
df = spark.sql(
    """
    select distinct vip_main_no, max(customer_age_group) customer_age_group, customer_tag from nj_sales group by vip_main_no, customer_tag
    """
)

table = count_pivot_table(df, group_by_col="customer_age_group", agg_col="vip_main_no").createOrReplaceTempView(
    "age_gp")

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
# MAGIC create or replace temp view nj_sales_first_purchase as
# MAGIC select nj_sales.vip_main_no, first_purchase.first_pur_cm, order_date, item_desc, maincat_desc_cleaned, maincat_and_subcat,new_joiner_flag, customer_tag from nj_sales
# MAGIC inner join first_purchase
# MAGIC on nj_sales.vip_main_no = first_purchase.vip_main_no
# MAGIC and first_purchase.first_pur_cm = nj_sales.order_date
# MAGIC where sales_staff_flag = 0
# MAGIC and new_joiner_flag = 1

# COMMAND ----------

# MAGIC %sql select * from nj_sales_first_purchase

# COMMAND ----------

def pivot_table_by_cat_1(group_by="item_subcat_desc_cleaned", agg_col="net_amt_hkd", mode="sum",
                       table="nj_sales_first_purchase"):
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
            FOR customer_tag IN (
                'Prime Connoisseur',
                'Menswear Specialist',
                'Outerwear Fashionista',
                'Bottoms and Dresses Diva',
                'Average Fashion Connoisseur')
            )
        """
    )
    display(df)
    return df


# COMMAND ----------

pivot_table_by_cat_1(group_by="maincat_desc_cleaned", agg_col="vip_main_no", mode="count")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW nj_first_purchase_1 AS
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     vip_main_no, 
# MAGIC     first_pur_cm,
# MAGIC     maincat_desc_cleaned,
# MAGIC     item_subcat_desc_cleaned,
# MAGIC     maincat_and_subcat,
# MAGIC     persona,
# MAGIC     item_desc,
# MAGIC     new_joiner_flag
# MAGIC   FROM 
# MAGIC     new_joiner 
# MAGIC   INNER JOIN 
# MAGIC     final_sales_table 
# MAGIC   USING (vip_main_no)
# MAGIC ) subquery
# MAGIC WHERE 
# MAGIC   subquery.new_joiner_flag = 1;

# COMMAND ----------

group_by="maincat_desc_cleaned"
agg_col="distinct vip_main_no"
mode="count"
table="nj_first_purchase_1"
df = spark.sql(
    f"""
    select * from
        (select 
            distinct 
            case when isnull({group_by}) = 1 or {group_by} = "N/A" then "Unknown" else {group_by} end as {group_by},
            persona, 
            {mode}({agg_col}) as overall_amount
        from {table}
        group by 
            persona,
            {group_by}
        )
        PIVOT (
        SUM(overall_amount)
        FOR persona IN (
            'Prime Connoisseur',
            'Menswear Specialist',
            'Outerwear Fashionista',
            'Bottoms and Dresses Diva',
            'Average Fashion Connoisseur')
        )
    """
)
display(df)

# COMMAND ----------

group_by="maincat_and_subcat"
agg_col="distinct vip_main_no"
mode="count"
table="nj_first_purchase_1"
df = spark.sql(
    f"""
    select * from
        (select 
            distinct 
            case when isnull({group_by}) = 1 or {group_by} = "N/A" then "Unknown" else {group_by} end as {group_by},
            persona, 
            {mode}({agg_col}) as overall_amount
        from {table}
        group by 
            persona,
            {group_by}
        )
        PIVOT (
        SUM(overall_amount)
        FOR persona IN (
            'Prime Connoisseur',
            'Menswear Specialist',
            'Outerwear Fashionista',
            'Bottoms and Dresses Diva',
            'Average Fashion Connoisseur')
        )
    """
)
display(df)

# COMMAND ----------

group_by="item_desc"
agg_col="distinct vip_main_no"
mode="count"
table="nj_first_purchase_1"
df = spark.sql(
    f"""
    select * from
        (select 
            distinct 
            case when isnull({group_by}) = 1 or {group_by} = "N/A" then "Unknown" else {group_by} end as {group_by},
            persona, 
            {mode}({agg_col}) as overall_amount
        from {table}
        group by 
            persona,
            {group_by}
        )
        PIVOT (
        SUM(overall_amount)
        FOR persona IN (
            'Prime Connoisseur',
            'Menswear Specialist',
            'Outerwear Fashionista',
            'Bottoms and Dresses Diva',
            'Average Fashion Connoisseur')
        )
    """
)
display(df)
