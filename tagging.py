# Databricks notebook source
import os

base_dir = "/mnt/dev/customer_segmentation/imx/club_monaco/datamart"

sales = spark.read.parquet(os.path.join(base_dir, "transaction.parquet"))
sales.createOrReplaceTempView("sales")
item_attr = spark.sql("SELECT DISTINCT item_desc, maincat_desc, item_subcat_desc FROM sales").toPandas()


# COMMAND ----------

def tagging(desc):
    tags = []
    for k, v in keywords.items():
        for keyword in v:
            if keyword in desc:
                tags.append(k.upper())
    return tags


keywords = {
    "stripe_pattern": ["stripe"],
    "non_stripe_pattern": ["floral", "solid", "jacquard", "ribbed", "plaid", "print"],
    "silk_fabric": ["silk"],
    "cashmere_fabric": ["cashmere"],
    "linen_fabric": ["linen"],
    "other_fabric": ["wool", "merino", "twill", "flannel", "lace", "oxford", "tweed", "pique", "boucle"],
    "cropped_style": ["crop", "cropped"],
    "textured_style": ["textured"],
    "mini_size": ["mini"],
    "short_size": ["short"],
    "other_size": ["maxi", "wide", "midi", "long"],
}

item_attr["tags"] = item_attr["item_desc"].apply(lambda x: tagging(x.lower()) if x is not None else [])
spark.createDataFrame(item_attr).write.parquet(os.path.join(base_dir, "item_attr_tagging.parquet"), mode="overwrite")
