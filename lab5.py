
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

# Create SparkSession
spark = SparkSession.builder \
    .appName("BankChurnMLlib") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.debug.maxToStringFields", "200") \
    .getOrCreate()

df = spark.read.csv('C:/Users/mrbre/Downloads/users-score-2023.csv', header=True, inferSchema=True, sep=",")

num_partitions = df.rdd.getNumPartitions()
print(f"Liczba partycji: {num_partitions}")

df_repartitioned = df.repartition(16)
df_repartitioned.write.csv("C:/Users/mrbre/Downloads/users-score-output.csv")

df_coalesced = df.coalesce(4)
df_coalesced.write.csv("C:/Users/mrbre/Downloads/coalesced_output.csv")

df_grouped = df.groupBy("Anime Title").count()
df_grouped.show()

df.persist()
df_grouped = df.groupBy("Anime Title").count()
df_grouped.show()

df_partitioned = df.repartition("Anime Title")
df_grouped = df_partitioned.groupBy("Anime Title").count()
df_grouped.show()

df.unpersist()

broadcast_data = spark.sparkContext.broadcast([1, 2, 3])
df_filtered = df.filter(df["Anime Title"].isin(broadcast_data.value))
df_filtered.show()

error_acc = spark.sparkContext.accumulator(0)

def process_row(row):
    if "error" in row:
        error_acc.add(1)

df.foreach(process_row)
print(f"Liczba błędów: {error_acc.value}")



