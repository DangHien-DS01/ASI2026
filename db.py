from sqlalchemy import create_engine

engine = create_engine("sqlite:///vn30.db")

def save_to_db(df):
    df.to_sql("vn30_prices", engine, if_exists="replace", index=False)
