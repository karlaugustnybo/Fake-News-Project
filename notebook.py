import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _(test):
    test
    return


if __name__ == "__main__":
    app.run()
