import yfinance as yf
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, desc, to_date

import matplotlib.pyplot as plt

# ============================================================
# LAYER 1: EXTRACT
# ============================================================

class NiftyExtractor:

    @staticmethod
    def extract(start_date: str) -> pd.DataFrame:
        end_date = datetime.today().strftime("%Y-%m-%d")

        pdf = yf.download("^NSEI", start=start_date, end=end_date)

        pdf.columns = pdf.columns.get_level_values(0)
        pdf.reset_index(inplace=True)
        pdf.columns.name = None

        return pdf

# ============================================================
# LAYER 2: TRANSFORM
# ============================================================

class NiftyTransformer:
    
    @staticmethod
    def transform(pdf: pd.DataFrame):
        spark = SparkSession.builder.getOrCreate()

        df = spark.createDataFrame(pdf)

        df = (
            df.withColumn("Date", to_date("Date"))
              .withColumn("Open", col("Open").cast("double"))
              .withColumn("Close", col("Close").cast("double"))
              .withColumn("High", col("High").cast("double"))
              .withColumn("Low", col("Low").cast("double"))
        )

        df = df.withColumn(
            "Change(%)",
            round(((col("Close") - col("Open")) / col("Open")) * 100, 2)
        )

        df = df.dropna(subset=["Change(%)"])

        return df

# ============================================================
# LAYER 3: LOAD (PERSISTENCE ONLY)
# ============================================================

class NiftyLoader:
   
    @staticmethod
    def write_csv(df, output_path: str):
        (
            df.orderBy("Date")
              .coalesce(1)
              .write
              .mode("overwrite")
              .option("header", True)
              .csv(output_path)
        )

        print(f"Data successfully loaded to {output_path}")

# ============================================================
# LAYER 4: ANALYSIS / CONSUMPTION
# ============================================================

class NiftyAnalytics:
    
    @staticmethod
    def read_csv(path: str):
        spark = SparkSession.builder.getOrCreate()
        return spark.read.option("header", True).csv(path, inferSchema=True)

    @staticmethod
    def show_analytics(df):
        print("Highest point where NIFTY 50 reached")
        df.orderBy(desc("High")).limit(1).show(truncate=False)

        print("Top 5 biggest Rise in NIFTY 50")
        df.orderBy(col("Change(%)").desc()).limit(5).show(truncate=False)

        print("Top 5 biggest Drop in NIFTY 50")
        df.orderBy(col("Change(%)").asc()).limit(5).show(truncate=False)

    @staticmethod
    def visualize(df):
        pdf = df.orderBy("Date").toPandas()

        plt.figure(figsize=(10, 6))
        for i in range(1, len(pdf)):
            color = "green" if pdf["Close"].iloc[i] >= pdf["Close"].iloc[i - 1] else "red"
            plt.plot(
                pdf["Date"].iloc[i-1:i+1],
                pdf["Close"].iloc[i-1:i+1],
                color=color
            )

        plt.title("NIFTY 50 Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ============================================================
# LAYER 5: PIPELINE ORCHESTRATION
# ============================================================

class NiftyETLPipeline:
   
    def __init__(self, start_date: str):
        self.start_date = start_date
        self.output_path = "/content/nifty50_csv"

    def run_etl(self):
        raw_pdf = NiftyExtractor.extract(self.start_date)
        transformed_df = NiftyTransformer.transform(raw_pdf)

        # ETL ENDS HERE
        NiftyLoader.write_csv(transformed_df, self.output_path)

        return self.output_path

# ============================================================
# ENTRY POINTS
# ============================================================

if __name__ == "__main__":
    pipeline = NiftyETLPipeline(start_date="2026-01-01")
    storage_path = pipeline.run_etl()

    # Downstream analytics (separate from ETL)
    analytics_df = NiftyAnalytics.read_csv(storage_path)
    NiftyAnalytics.show_analytics(analytics_df)
    NiftyAnalytics.visualize(analytics_df)

