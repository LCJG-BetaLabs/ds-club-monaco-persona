# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("start_date", "")
dbutils.widgets.text("end_date", "")
dbutils.widgets.text("base_dir", "")

# COMMAND ----------

import os

datamart_dir = dbutils.widgets.get("base_dir") + "/datamart"

sales = spark.read.parquet(os.path.join(datamart_dir, "transaction.parquet"))
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

def tagging_1(desc, current_tags):
    tags = current_tags.copy()
    for k, v in keywords.items():
        for keyword in v:
            if keyword in desc and k.upper() not in tags:
                tags.append(k.upper())
    return tags

# COMMAND ----------

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

# COMMAND ----------

keywords = {
    "mw": ["m_outerwear/blazers", "m_bottoms", "m_knit tops", "m_sweaters", "m_shirts"],
    "ww": ["w_outerwear/blazers", "w_bottoms", "w_sweaters", "w_shirts", "w_skirts", "w_dresses", "w_knit tops"],
    "top": ["m_knit tops", "w_knit tops", "m_shirts", "w_shirts", "w_dresses"],
    "bottom": ["m_bottoms", "w_bottoms", "w_skirts"],
    "outerwear": ["m_outerwear/blazers", "w_outerwear/blazers", "m_sweaters", "w_sweaters"],
    "accessories": ["mens accessories", "womens accessories", "home accessories"]
}

item_attr["tags"] = item_attr.apply(lambda row: tagging_1(row["maincat_desc"].lower(), row["tags"]) if row["maincat_desc"] is not None else row["tags"], axis=1)
spark.createDataFrame(item_attr).write.parquet(os.path.join(datamart_dir, "item_attr_tagging.parquet"), mode="overwrite")

# COMMAND ----------

# another tagging parquet for profiling
keywords = {
    "mw_top": ["m_knit tops", "m_shirts"],
    "ww_top": ["w_knit tops", "w_shirts", "w_dresses"],
    "mw_bottom": ["m_bottoms"],
    "ww_bottom": ["w_bottoms", "w_skirts"],
    "mw_outerwear": ["m_outerwear/blazers", "m_sweaters"],
    "ww_outerwear": ["w_outerwear/blazers", "w_sweaters"],
    "accessories": ["mens accessories", "womens accessories", "home accessories"]
}

item_attr["tags"] = item_attr["maincat_desc"].apply(lambda x: tagging(x.lower()) if x is not None else [])
spark.createDataFrame(item_attr).write.parquet(os.path.join(datamart_dir, "item_attr_tagging1.parquet"), mode="overwrite")
