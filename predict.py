# Databricks notebook source
# MAGIC %py
# MAGIC dbutils.widgets.removeAll()
# MAGIC dbutils.widgets.text("start_date", "")
# MAGIC dbutils.widgets.text("end_date", "") 
# MAGIC dbutils.widgets.text("base_dir", "")

# COMMAND ----------

import os
import glob
import numpy as np
import pandas as pd
import pyspark.sql.functions as f
import joblib

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE  

import matplotlib.pyplot as plt

# COMMAND ----------

feature_dir = os.path.join(dbutils.widgets.get("base_dir"), "features")
model_dir = os.path.join(dbutils.widgets.get("base_dir"), "model")
os.makedirs(model_dir, exist_ok=True)
files = glob.glob("/dbfs" + feature_dir + "/*.parquet")

features_df = None
for path in files:
    if "demographic" in path or "transactional" in path:
        continue
    if "tagging.parquet" not in path:
        df = spark.read.parquet(path.replace("/dbfs", ""))
        if features_df is None:
            features_df = df
        else:
            features_df = features_df.join(df, on='vip_main_no', how='inner')
    else:
        df = spark.read.parquet(path.replace("/dbfs", ""))
        features_df = features_df.join(df, on='vip_main_no', how='left')

# COMMAND ----------

features_df = features_df.fillna(0)
features_to_keep = ['vip_main_no',
 'brand_CM_qty',
 'maincat_W_BOTTOMS_qty',
 'maincat_W_DRESSES_qty',
 'maincat_W_KNIT TOPS_qty',
 'maincat_W_OUTERWEAR/BLAZERS_qty',
 'maincat_W_SHIRTS_qty',
 'maincat_W_SKIRTS_qty',
 'maincat_W_SWEATERS_qty',
 'subcat_3rd PARTY_qty',
 'subcat_W_FULL LENGTH PANTS_qty',
 'subcat_W_WOVEN DRESSES_qty',
 'subcat_W_WOVEN SKIRTS_qty',
 'tag_BOTTOM_qty',
 'tag_MW_qty',
 'tag_NON_STRIPE_PATTERN_qty',
 'tag_OTHER_FABRIC_qty',
 'tag_OUTERWEAR_qty',
 'tag_TOP_qty',
 'tag_WW_qty']
features_df_filtered = features_df.select(features_to_keep)

# COMMAND ----------

feature_cols = [c for c in features_df_filtered.columns if c != "vip_main_no"]
all_vip = features_df_filtered.select("vip_main_no").toPandas().values.reshape(1, -1)

# COMMAND ----------

# get feature array
pandas_df = features_df_filtered.select(feature_cols).toPandas()
features_array = pandas_df.values

# COMMAND ----------

# standardization
scaler = StandardScaler()
standardized_df = scaler.fit_transform(features_array)
standardized_df = np.nan_to_num(standardized_df)

# COMMAND ----------

selected_features = [14,  0, 13, 18, 17, 12, 15,  9,  8, 16]

# COMMAND ----------

# MAGIC %md
# MAGIC selected feature:
# MAGIC ```python
# MAGIC array(['tag_NON_STRIPE_PATTERN_qty', 'brand_CM_qty', 'tag_MW_qty',
# MAGIC        'tag_WW_qty', 'tag_TOP_qty', 'tag_BOTTOM_qty',
# MAGIC        'tag_OTHER_FABRIC_qty', 'subcat_W_FULL LENGTH PANTS_qty',
# MAGIC        'subcat_3rd PARTY_qty', 'tag_OUTERWEAR_qty'], dtype='<U31')
# MAGIC ```

# COMMAND ----------

features_embed = standardized_df[:, selected_features]

# COMMAND ----------

import joblib
from sklearn.cluster import KMeans
kmeans = joblib.load("/dbfs/mnt/prd/customer_segmentation/imx/club_monaco/train/model/kmeans_model3.pkl")
cluster_assignment = kmeans.predict(features_embed)

# COMMAND ----------

result_df = pd.DataFrame(np.concatenate((all_vip.reshape(-1, 1), cluster_assignment.reshape(-1, 1)), axis=1),
                         columns=["vip_main_no", "persona"])
spark.createDataFrame(result_df).write.parquet(os.path.join(model_dir, "clustering_result.parquet"), mode="overwrite")

# COMMAND ----------

subfeatures_embed = features_embed[cluster_assignment == 2]
sub_kmeans = joblib.load(os.path.join("/dbfs/mnt/prd/customer_segmentation/imx/club_monaco/train/model/sub_kmeans_model3.pkl"))
subcluster_assignment = sub_kmeans.predict(subfeatures_embed)

# COMMAND ----------

subset_all_vip = result_df[result_df['persona'] == 2]['vip_main_no'].values
sub_result_df = pd.DataFrame(np.concatenate((subset_all_vip.reshape(-1, 1), subcluster_assignment.reshape(-1, 1)), axis=1),
                         columns=["vip_main_no", "persona"])

# COMMAND ----------

# save result
spark.createDataFrame(sub_result_df).write.parquet(os.path.join(model_dir, "clustering_sub_result.parquet"), mode="overwrite")
