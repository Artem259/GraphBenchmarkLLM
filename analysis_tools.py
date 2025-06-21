import pandas as pd
import json
import re


def load_data(file: str) -> list[pd.DataFrame]:
    with open(file, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df["correct"] = df["correct"].astype(bool)
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")

    dfs = [v for _, v in df.groupby("num_examples", sort=False)]
    dfs.sort(key=lambda df_: df_["num_examples"].iloc[0])
    return dfs


def gen_count_table(df: pd.DataFrame, columns: list[str], rows: list[str]) -> pd.DataFrame:
    count_table = (
        df.groupby(["task", "model"], sort=False)["correct"]
        .count()
        .unstack()
    )
    count_table = count_table.reindex(columns=columns, fill_value=0)
    count_table = count_table.loc[rows]

    return count_table


def gen_accuracy_table(df: pd.DataFrame, columns: list[str], rows: list[str]) -> pd.DataFrame:
    accuracy_table = (
        df.groupby(["task", "model"], sort=False)["correct"]
        .mean()
        .unstack()
        .mul(100)
        .round(1)
    )
    accuracy_table = accuracy_table.reindex(columns=columns, fill_value=0)
    accuracy_table = accuracy_table.loc[rows]

    return accuracy_table


def gen_accuracy_model_table(df: pd.DataFrame, columns: list[str], rows: list[str]) -> pd.DataFrame:
    accuracy_table = gen_accuracy_table(df, columns, rows)
    accuracy_model_table = accuracy_table.mean().round(1).to_frame()

    return accuracy_model_table


def gen_accuracy_task_table(df: pd.DataFrame, columns: list[str], rows: list[str]) -> pd.DataFrame:
    accuracy_table = gen_accuracy_table(df, columns, rows)
    accuracy_task_table = accuracy_table.mean(axis=1).round(1).to_frame()

    return accuracy_task_table


def gen_accuracy_task_tables(df: pd.DataFrame, tasks: list[str], rows: list[str]):
    accuracy_task_tables = {}
    for task, df_v in df.groupby("task", sort=False):
        accuracy_task_table = (
            df_v.groupby(["model", "prompt"], sort=False)["correct"].mean().unstack().round(0).fillna(-1).astype(int)
        )
        accuracy_task_table = accuracy_task_table.loc[rows]

        for prompt in accuracy_task_table.columns:
            matches = re.findall(r"\[\(\d+, \d+(?:\), \(\d+, \d+)*\)]", prompt)
            if matches:
                edges_list_str = matches[-1]
                edges_list = eval(edges_list_str)
                edges_num = len(edges_list)
                accuracy_task_table = accuracy_task_table.rename(columns={prompt: edges_num})
            else:
                raise ValueError("Bad prompt.")

        accuracy_task_tables[task] = accuracy_task_table
    accuracy_task_tables = {task: accuracy_task_tables[task] for task in tasks if task in accuracy_task_tables}

    return accuracy_task_tables


def gen_time_table(df: pd.DataFrame, columns: list[str], rows: list[str]) -> pd.DataFrame:
    time_table = (
        df.groupby(["task", "model"], sort=False)["time_sec"]
        .mean()
        .unstack()
        .round(1)
    )
    time_table = time_table.reindex(columns=columns, fill_value=0)
    time_table = time_table.loc[rows]

    return time_table


def gen_time_model_table(df: pd.DataFrame, columns: list[str], rows: list[str]) -> pd.DataFrame:
    time_table = gen_time_table(df, columns, rows)
    time_model_table = time_table.mean().round(1).to_frame()

    return time_model_table


