import duckdb

con = duckdb.connect("data/silver/retail_silver.duckdb", read_only=True)
con.sql("""
    SELECT date, dayname(date) AS weekday, units
    FROM silver_sales
    WHERE id = 'FOODS_3_090_CA_3_evaluation'
      AND date IN ('2016-03-21', '2015-04-20', '2016-03-27', '2015-04-26')
    ORDER BY date
""").show()
con.close()
