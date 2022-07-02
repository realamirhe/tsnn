import pandas as pd
import plotly.express as px

if __name__ == "__main__":
    df = pd.read_csv("../../out/csv/dst firing-1.csv")
    df = df.astype(int)
    # fig = px.line(df[["id", "abc"]], x="id", y="abc", title="abc")
    # fig = px.line(df[["id", "omn"]], x="id", y="omn", title="omn")
    fig = px.scatter(df[["abc", "omn"]])
    # fig = px.scatter(df[["id", "abc"]], x="id", y="abc", title="abc", color=1)
    # fig.show()
    fig.show()
