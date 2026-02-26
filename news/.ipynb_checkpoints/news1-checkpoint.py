import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <br><h1 style='text-align: center;'>Fake News</h1><br>
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nltk
    import re

    pd.set_option('display.max_columns', 300)
    plt.style.use('ggplot')
    return (pd,)


@app.cell
def _(pd):
    filepath = './data/news_sample.csv'

    df = pd.read_csv(filepath, index_col='id').iloc[:, 1:]
    df
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _():
    df.drop(['keywords','summary'], axis=1, inplace=True)
    df = df.fillna('nan')
    df['meta_keywords'] = df['meta_keywords'].replace("['']",'nan')
    df
    return (df,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
