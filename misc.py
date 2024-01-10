# Databricks notebook source
import os

base_dir = "/mnt/dev/customer_segmentation/imx/club_monaco/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
vip = spark.read.parquet(os.path.join(base_dir, "demographic.parquet"))
first_purchase = spark.read.parquet(os.path.join(base_dir, "first_purchase.parquet"))

# COMMAND ----------

sales.createOrReplaceTempView("sales")
vip.createOrReplaceTempView("vip")
first_purchase.createOrReplaceTempView("first_purchase")

# COMMAND ----------

# get item list
def get_item_list(save=False):
    df = spark.sql("SELECT DISTINCT prod_brand, item_desc, maincat_desc, item_subcat_desc FROM sales").toPandas()
    if save:
        df.to_csv("/dbfs/mnt/dev/customer_segmentation/imx/club_monaco/item_list.csv", index=False)
    return df

# COMMAND ----------

def get_brand_list():
    df = spark.sql(
        """
        select
            distinct prod_brand,
            brand_desc
        from
            CMSalesProduct a
        left join imx_prd.imx_dw_train_silver.dbo_viw_lc_xxx_brand_brand b on prod_brand = b.brand_code
        """
    )
    return df
