import os
import sys
from argparse import Namespace
from datetime import date, timedelta
from typing import Optional

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from spark_datascience.tables.eventlog import Eventlog
from spark_datascience.output_dataset import OutputDataset

spark: SparkSession = SparkSession.builder.getOrCreate()


eventlog_table = Eventlog()

UA_MAP_PATH: str = "s3://s3-data-analysis-usw2-prd/eventlog-user-agent-map"

start_date = "2020-10-01"
end_date = "2020-11-30"

unproxied_frame: DataFrame = spark.read.parquet(UA_MAP_PATH).filter(
    F.col("dt").between(start_date, end_date) & ~F.col("is_proxy")
).select(
    "dt",
    "event_id",
    "agent_family",
    "device_model",
    "device_brand",
    "is_mobile"
).withColumn(
    "device_info",
    F.struct(F.col("agent_family"), F.col("device_brand"), F.col("device_model"), F.col("is_mobile"))
)

eventlog_frame = eventlog_table.get_frame(
    partition_prefix="2020-11-*"
).filter(
    F.col("tenant").isin(["spc"]) &
    F.col("dt").between(start_date, end_date) &
#   ~F.col("geo_ip_country").isNull() &
    F.col("type").isin(["open", "is_open", "initial_open", "click", "amp_click"])
).select(
    "dt",
    "tenant",
    "customer_id",
    "event_id",
    "rcpt_to_hash",
    "geo_ip_country",
    "geo_ip_city"
).withColumn("geo_pair", F.struct(F.col("geo_ip_country"), F.col("geo_ip_city")))

geo_frame = eventlog_frame.join(unproxied_frame, on=["event_id", "dt"], how="inner").groupby(
    "tenant",
    "customer_id",
    "rcpt_to_hash"
).agg(
    F.collect_set("geo_pair").alias("geo_info"),
    F.collect_set("device_info").alias("device_info")
)




engagement_frame = eventlog_table.get_frame(
    partition_prefix="2020-*"
).filter(
    F.col("tenant").isin(["spc"]) &
    F.col("dt").between(start_date, end_date) &
    F.col("type").isin(["open", "amp_open", "initial_open", "click", "amp_click"])
).select(
    "dt",
    "tenant",
    "customer_id",
    "rcpt_to_hash",
    "type"
).groupby(
    "tenant",
    "customer_id",
    "rcpt_to_hash"
).agg(
    F.max(F.col("dt")).alias("last_engagement_1p"),
    F.sum(F.when(F.col("type").isin(["open", "amp_open", "amp_initial_open", "initial_open"]),1).otherwise(0)).alias("opens_1p"),
    F.sum(F.when(F.col("type").isin(["click", "amp_click"]),1).otherwise(0)).alias("clicks_1p"),
    F.sum(F.when(F.col("type").isin(["amp_open", "amp_initial_open" "amp_click"]),1).otherwise(0)).alias("amp_1p")
)

complaint_frame = eventlog_table.get_frame(
    partition_prefix="2020-*"
).filter(
    F.col("tenant").isin(["spc"]) &
    F.col("dt").between(start_date, end_date) &
    F.col("type").isin(["complaint", "link_unsubscribe", "list_unsubscribe"])
).select(
    "dt",
    "tenant",
    "customer_id",
    "rcpt_to_hash",
    "type"
).groupby(
    "tenant",
    "customer_id",
    "rcpt_to_hash"
).agg(
    F.sum(F.when(F.col("type").isin(["complaint"]),1).otherwise(0)).alias("complaints_1p"),
    F.sum(F.when(F.col("type").isin(["list_unsubscribe", "link_unsubscribe"]),1).otherwise(0)).alias("unsubs_1p")
)

d_frame = eventlog_table.get_frame(
    partition_prefix="2020-*"
).filter(
    F.col("tenant").isin(["spc"]) &
    F.col("dt").between(start_date, end_date) &
    F.col("type").isin(["delivery", "bounce"])
).select(
    "dt",
    "tenant",
    "customer_id",
    "rcpt_to_hash",
    "type",
    "bounce_class"
).groupby(
    "tenant",
    "customer_id",
    "rcpt_to_hash"
).agg(
    F.max(F.col("dt")).alias("last_attempt_1p"),
    F.sum(F.when(F.col("type") == "delivery",1).otherwise(0)).alias("deliveries_1p"),
    F.sum(F.when((F.col("type") == "bounce") & (F.col("bounce_class") == 10),1).otherwise(0)).alias("hardbounce_1p")
)

df = d_frame.join(engagement_frame, on=["tenant", "customer_id", "rcpt_to_hash"], how="outer")
df = df.join(geo_frame, on=["tenant", "customer_id", "rcpt_to_hash"], how="outer")
df = df.join(complaint_frame, on=["tenant", "customer_id", "rcpt_to_hash"], how="outer")

output_dataset = OutputDataset(
    df,
    output_path="s3://s3-data-analysis-usw2-prd/gs/team-wookie/1p-insights-2",
    auto_repartition=True,
    rows_per_file=50000000
)
output_dataset.write(overwrite=True, set_acl=True)
