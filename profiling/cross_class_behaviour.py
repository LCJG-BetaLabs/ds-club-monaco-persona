# Databricks notebook source
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

import pandas as pd


def get_cross_class_pivot_table(
    persona="Average Fashion Connoisseur",
    _class="maincat_desc_cleaned",
    aggfunc=lambda x: len(x.unique()),
    values="vip_main_no",
):
    table = spark.sql(
        f"""select 
            vip_main_no,  
            sold_qty,
            net_amt_hkd,
            sales_main_key,
            order_date,
            maincat_desc_cleaned,
            maincat_and_subcat
        from final_sales_table 
        where persona = '{persona}'
    """
    ).toPandas()
    table_outer = table.merge(table, how="outer", on="vip_main_no")
    if aggfunc == "sum":
        table_outer[values] = table_outer.apply(lambda row: int(row[f"{values}_x"]) +int(row[f"{values}_y"]), axis=1)
    pivot_table = pd.pivot_table(
        table_outer,
        values=values,
        index=f"{_class}_x",
        columns=f"{_class}_y",
        aggfunc=aggfunc,
        fill_value=0,
        margins=True,
    )
    return pivot_table

# COMMAND ----------

# vip count
for i in range(5):
    df = get_cross_class_pivot_table(
        persona=cluster_order[i],
        _class="maincat_desc_cleaned",
        aggfunc=lambda x: len(x.unique()),
        values="vip_main_no",
    )
    display(df)

# COMMAND ----------

# vip count
for i in range(5):
    df = get_cross_class_pivot_table(
        persona=cluster_order[i],
        _class="maincat_and_subcat",
        aggfunc=lambda x: len(x.unique()),
        values="vip_main_no",
    )
    display(df)

# COMMAND ----------

# amt
for i in range(5):
    df = get_cross_class_pivot_table(
        persona=cluster_order[i],
        _class="maincat_desc_cleaned",
        aggfunc="sum",
        values="net_amt_hkd",
    )
    display(df)

# COMMAND ----------

# for tagging
item_attr = spark.read.parquet(os.path.join(base_dir, "item_attr_tagging1.parquet")).toPandas()
item_attr_exploded = item_attr.explode("tags")
item_attr_exploded_spark = spark.createDataFrame(item_attr_exploded)
final_sales_table_with_tags = final_sales_table.join(item_attr_exploded_spark.select("item_desc", "tags"),
                                                     on="item_desc", how="left")

# COMMAND ----------

final_sales_table_with_tags.createOrReplaceTempView("final_sales_table_with_tags")

# COMMAND ----------

def get_cross_class_pivot_table_1(
    persona="Average Fashion Connoisseur",
    _class="tags",
    aggfunc=lambda x: len(x.unique()),
    values="vip_main_no",
):
    table = spark.sql(
        f"""select 
            vip_main_no,  
            sold_qty,
            net_amt_hkd,
            sales_main_key,
            order_date,
            maincat_desc_cleaned,
            maincat_and_subcat,
            tags
        from final_sales_table_with_tags 
        where persona = '{persona}'
    """
    ).toPandas()
    table_outer = table.merge(table, how="outer", on="vip_main_no")
    if aggfunc == "sum":
        table_outer[values] = table_outer.apply(lambda row: int(row[f"{values}_x"]) +int(row[f"{values}_y"]), axis=1)
    pivot_table = pd.pivot_table(
        table_outer,
        values=values,
        index=f"{_class}_x",
        columns=f"{_class}_y",
        aggfunc=aggfunc,
        fill_value=0,
        margins=True,
    )
    return pivot_table

# COMMAND ----------

# vip count
for i in range(5):
    df = get_cross_class_pivot_table_1(
        persona=cluster_order[i],
        _class="tags",
        aggfunc=lambda x: len(x.unique()),
        values="vip_main_no",
    )
    display(df)
