from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)
import json


#Dataset:
#https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

def analyze_dataset_distribution(df):
    columns_to_exclude = ["row_number", "track_id", "artists", "album_name", "track_name"]

    distribution_analysis = {}
    for column in df.columns:
        if column in columns_to_exclude:
            continue

        col_data = {}
        if isinstance(df.schema[column].dataType, (IntegerType, FloatType)):
            stats = df.select(
                F.mean(column).alias("mean"),
                F.stddev(column).alias("stddev"),
                F.min(column).alias("min"),
                F.max(column).alias("max")
            ).collect()[0]
            percentiles = df.approxQuantile(column, [0.25, 0.5, 0.75], 0.05)
            col_data = {
                "mean": stats["mean"],
                "stddev": stats["stddev"],
                "min": stats["min"],
                "max": stats["max"],
                "25th_percentile": percentiles[0],
                "50th_percentile": percentiles[1],
                "75th_percentile": percentiles[2],
            }
        elif isinstance(df.schema[column].dataType, StringType):
            col_data = df.groupBy(column).count().orderBy("count", ascending=False).collect()
            col_data = {row[column]: row["count"] for row in col_data}
        
        col_data["missing_values"] = df.filter(F.col(column).isNull()).count()
        distribution_analysis[column] = col_data

    output_path = "distribution_analysis.json" 
    with open(output_path, "w") as f:
        json.dump(distribution_analysis, f, indent=4)
    print(f"Analysis saved to {output_path}")

def analyze_collaboration_popularity(df):
    
    collaboration_df = (
        df.filter(
            F.col("artists").contains(";")
        ).groupBy("artists").agg(
            F.avg("popularity").alias("avg_popularity"),
            F.count("*").alias("num_collabs")
        ).orderBy("num_collabs", ascending=False).limit(10)
    )

    artist_popularity_df = (
        df.select("artists", "popularity")
        .withColumn("artist", F.explode(F.split(F.col("artists"), ";")))
        .groupBy("artist")
        .agg(F.avg("popularity").alias("artist_avg_popularity"))
    )
    
    collab_with_least_popular_df = (
        collaboration_df
        .withColumn("artist", F.explode(F.split(F.col("artists"), ";")))  
        .join(artist_popularity_df, on="artist", how="left")  
        .groupBy("artists", "avg_popularity", "num_collabs")  
        .agg(F.min("artist_avg_popularity").alias("least_popular_artist_avg")) 
        .orderBy("num_collabs", ascending=False)
    )
                               
    return collab_with_least_popular_df

def analyze_breakthrough_songs(df):

    albums_df = (
        df.groupBy("album_name")
        .agg(
            F.avg("popularity").alias("album_popularity"),
            F.avg("energy").alias("album_energy"),
            F.avg("danceability").alias("album_danceability"),
            F.avg("valence").alias("album_valence")
        )
    )

    breakthrough_songs_df = (
        df.filter(
            (F.col("popularity") > 80)
            & (F.col("album_name").isNotNull())
        )
        .join(albums_df, on="album_name", how="left")
        .filter(F.col("album_popularity") < 50)
        .withColumn("energy_diff[%]", (F.col("energy") - F.col("album_energy")) / F.col("album_energy") * 100)
        .withColumn("danceability_diff[%]", (F.col("danceability") - F.col("album_danceability")) / F.col("album_danceability") * 100)
        .withColumn("valence_diff[%]", (F.col("valence") - F.col("album_valence")) / F.col("album_valence") * 100)
        .select(
            "track_name",
            "artists",
            "album_name",
            "popularity",
            "energy_diff[%]",
            "danceability_diff[%]",
            "valence_diff[%]"
        )
    )

    return breakthrough_songs_df

def analyze_genre_sweet_spot(df):
    genre_df = (
        df.groupBy("track_genre")
        .agg(
            F.count("*").alias("num_tracks"),
            F.avg(
                F.when(F.col("tempo") < 100.0, F.col("popularity"))
            ).alias("slow_tempo_popularity"),
            F.avg(
                F.when((F.col("tempo") >= 100.0) & (F.col("tempo") < 120.0), F.col("popularity"))
            ).alias("medium_tempo_popularity"),
            F.avg(
                F.when(F.col("tempo") >= 120.0, F.col("popularity"))
            ).alias("fast_tempo_popularity")
            )
        .orderBy("num_tracks", ascending=False)
        .limit(5)
    )
    
    return genre_df

def analyze_explicit_content_popularity_by_genre(df):
    genre_df = (
        df.groupBy("track_genre")
        .agg(
            F.avg(
                F.when(F.col("explicit") == "True", F.col("popularity"))
            ).alias("explicit_popularity"),
            F.avg(
                F.when(F.col("explicit") == "False", F.col("popularity"))
            ).alias("nonexplicit_popularity")
        )
        .withColumn(
            "difference",
            F.col("explicit_popularity") - F.col("nonexplicit_popularity")
        )
    )

    return genre_df

def analyze_popularity_by_length_and_danceability(df):
    longest_songs_df = (
        df.filter(F.col("danceability") > 0.8)
        .orderBy("duration_ms", ascending = False)
        .limit(10)
        .select(
            "track_name",
            "track_genre",
            "duration_ms",
            "danceability",
            "popularity"
        )
    )

    genre_df = (
        df.groupBy("track_genre")
        .agg(
            F.avg(F.col("popularity")).alias("genre_avg_popularity")
        )
    )

    result_df = (
        longest_songs_df.join(genre_df, on = "track_genre", how = "left")
        .withColumn(
            "popularity_diff",
            F.col("popularity") - F.col("genre_avg_popularity")
        )
    )
    
    return result_df

def analyze_explicit_valence_patterns(df):

    popularity_segmentation = [
        F.col("popularity") < 10,
        (F.col("popularity") >= 10) & (F.col("popularity") < 20),
        (F.col("popularity") >= 20) & (F.col("popularity") < 30),
        (F.col("popularity") >= 30) & (F.col("popularity") < 40),
        (F.col("popularity") >= 40) & (F.col("popularity") < 50),
        (F.col("popularity") >= 50) & (F.col("popularity") < 60),
        (F.col("popularity") >= 60) & (F.col("popularity") < 70),
        (F.col("popularity") >= 70) & (F.col("popularity") < 80),
        (F.col("popularity") >= 80) & (F.col("popularity") < 90),
        (F.col("popularity") >= 90),
    ]

    result_df = (
        df.select(
            "popularity",
            "explicit",
            "valence"
        )
        .withColumn(
            "popularity_range",
            F.when(popularity_segmentation[0], "0-10")
            .when(popularity_segmentation[1], "10-20")
            .when(popularity_segmentation[2], "20-30")
            .when(popularity_segmentation[3], "30-40")
            .when(popularity_segmentation[4], "40-50")
            .when(popularity_segmentation[5], "50-60")
            .when(popularity_segmentation[6], "60-70")
            .when(popularity_segmentation[7], "70-80")
            .when(popularity_segmentation[8], "80-90")
            .when(popularity_segmentation[9], "90-100")
        )
        .groupBy("popularity_range")
        .agg(
            F.avg(
                F.when(F.col("explicit") == "True", F.col("valence"))
            ).alias("explicit_valence"),
            F.avg(
                F.when(F.col("explicit") == "False", F.col("valence"))
            ).alias("nonexplicit_valence")
        ).withColumn(
            "difference[%]",
            (F.col("explicit_valence") - F.col("nonexplicit_valence")) / F.col("nonexplicit_valence") * 100
        ).orderBy("popularity_range")
    )

    return result_df

def analyze_artist_consistency(df):

    result_df = (
        df.withColumn(
            "artist", 
            F.explode(F.split(F.col("artists"), ";"))
        )
        .groupBy("artist","track_genre")
        .agg(
            F.count("*").alias("num_tracks"),
            F.avg(F.col("popularity")).alias("popularity_average"),
            F.stddev(F.col("popularity")).alias("popularity_deviation"),
        )
        #total number of tracks per artist
        .withColumn(
            "total_tracks",
            F.sum("num_tracks").over(Window.partitionBy("artist"))
        )
        .withColumn(
            "genre_percentage",
            F.col("num_tracks") / F.col("total_tracks")
        )
        .withColumn(
            "expected_value",
            F.sum(F.col("num_tracks") * F.col("genre_percentage")).over(Window.partitionBy("artist"))
        )
        .withColumn(
            "expected_value_squared",
            F.sum(F.pow(F.col("num_tracks"),2) * F.col("genre_percentage")).over(Window.partitionBy("artist"))
        )
        .withColumn(
            "variance",
            F.col("expected_value_squared") - F.pow(F.col("expected_value"),2)
        )
        .groupBy("artist")
        .agg(
            F.avg(F.col("popularity_average")).alias("popularity_average"),
            #replace null values with 0
            F.avg(F.when(F.col("popularity_deviation").isNull(),0).otherwise(F.col("popularity_deviation"))).alias("popularity_deviation"),
            F.avg(F.col("variance")).alias("variance")
        )
        .orderBy("popularity_deviation", ascending=False)
    )
    
    return result_df

def analyze_instrumental_impact(df):

    genre_df = (
        df.groupBy("track_genre")
        .agg(
            F.avg(F.col("acousticness")).alias("avg_acousticness"),
            F.avg(F.col("instrumentalness")).alias("avg_instrumentalness"),
            F.avg(F.col("popularity")).alias("avg_popularity")
        )
        .withColumn(
            "higly_instrumental",

            #if avg_acousticness or avg_instrumentalness then true othervwise false
            F.when(
                (F.col("avg_acousticness") > 0.8) | (F.col("avg_instrumentalness") > 0.8), True
            ).otherwise(False)
        )
        .groupBy("higly_instrumental")
        .agg(
            F.avg(F.col("avg_popularity")).alias("avg_popularity")
        )
    )

    return genre_df

def main():
    schema = StructType(
        [
            StructField("row_number", IntegerType(), True),
            StructField("track_id", StringType(), True),
            StructField("artists", StringType(), True),
            StructField("album_name", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("popularity", IntegerType(), True),
            StructField("duration_ms", IntegerType(), True),
            StructField("explicit", StringType(), True),
            StructField("danceability", FloatType(), True),
            StructField("energy", FloatType(), True),
            StructField("key", IntegerType(), True),
            StructField("loudness", FloatType(), True),
            StructField("mode", IntegerType(), True),
            StructField("speechiness", FloatType(), True),
            StructField("acousticness", FloatType(), True),
            StructField("instrumentalness", FloatType(), True),
            StructField("liveness", FloatType(), True),
            StructField("valence", FloatType(), True),
            StructField("tempo", FloatType(), True),
            StructField("time_signature", IntegerType(), True),
            StructField("track_genre", StringType(), True)
        ]
    )
    
    spark = SparkSession.builder \
    .appName("SpotifyAnalasys") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .getOrCreate()

    df = spark.read.csv(
        "dataset.csv",
        schema=schema,
        header=True,
        quote='"',  # Defines the quote character (default is double quotes)
        escape='"',  # Escape embedded quotes within quoted fields
        multiLine=True  # Allows fields to span multiple lines if necessary
    )

    analyze_dataset_distribution(df)
    print("Collaboration popularity: ")
    analyze_collaboration_popularity(df).show()
    print("Breakthrough songs: ")
    analyze_breakthrough_songs(df).show()
    print("Genre sweet spot: ")
    analyze_genre_sweet_spot(df).show()
    print("Explicit content popularity by genre: ")
    analyze_explicit_content_popularity_by_genre(df).show()
    print("Popularity by length and danceability: ")
    analyze_popularity_by_length_and_danceability(df).show()
    print("Explicit valence patterns: ")
    analyze_explicit_valence_patterns(df).show()
    print("Artist consistency: ")
    analyze_artist_consistency(df).show()
    print("Instrumental impact: ")
    analyze_instrumental_impact(df).show()

    spark.stop()


if __name__ == "__main__":
    main()