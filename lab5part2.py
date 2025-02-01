from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

spark = SparkSession.builder \
    .appName("LargeTextDataProcessing") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

text_df = spark.read.text("C:/Users/mrbre/Downloads/en_US.blogs.txt")

stopwords = ["a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "is", "it", "of"]
broadcast_stopwords = spark.sparkContext.broadcast(stopwords)

word_counts = text_df.rdd \
    .flatMap(lambda row: row[0].lower().split(" ")) \
    .filter(lambda word: word not in broadcast_stopwords.value and word.isalpha()) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda a, b: a + b)

word_counts.persist(StorageLevel.MEMORY_AND_DISK)

optimized_word_counts = word_counts.repartition(8)

output_path = "C:/Users/mrbre/Downloads/output.txt"
optimized_word_counts.saveAsTextFile(output_path)

print(f"Liczba unikalnych słów: {optimized_word_counts.count()}")

word_counts.unpersist()

spark.stop()
