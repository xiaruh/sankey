"""
filename: sankey.py
description: reusable library for sankey diagram
"""

import pandas as pd
import plotly.graph_objects as go

def map_columns_to_numbers(df, src, targ):
    """
    Map source and target columns to numbers for Sankey Diagram.
    Args:
        df (DataFrame): Input DataFrame.
        src (str): Source column name.
        targ (str): Target column name.
    Returns:
        df (DataFrame): Mapped DataFrame with numeric columns.
        labels (list): List of distinct labels for nodes.
    """

    # Get distinct labels
    labels = sorted(list(set(list(df[src]) + list(df[targ]))))

    # Get integer codes
    codes = list(range(len(labels)))

    # Create label to code mapping
    lc_map = dict(zip(labels, codes))

    # Substitute names for codes in dataframe
    df = df.replace({src: lc_map, targ: lc_map})

    return df, labels

def stack_columns_to_dataframe(df, *cols, vals=None):
    """
    Stack columns to create a concatenated DataFrame for Sankey Diagram.
    Args:
        df (DataFrame): Input DataFrame.
        cols (tuple): Columns used as source or target.
        vals (str): Column name for values (counts).
    Returns:
        stacked (DataFrame): Stacked DataFrame with source, target, and count columns.
    """

    # Create pairs from columns to have one src and one targ at a time
    pairs = list(zip(cols, cols[1:]))

    # If there's no values column
    if vals is None:
        stacked = pd.DataFrame({'src': [], 'targ': []})
        for src, targ in pairs:
            # Aggregate the data, counting the number of items
            grouped = df[[src, targ]].groupby([src, targ]).size().reset_index()
            grouped.columns = ['src', 'targ', 'num']
            # Concatenate df with only one pair to the current df
            stacked = pd.concat([stacked, grouped], axis=0)

    # If there's a values column
    else:
        stacked = pd.DataFrame({'src': [], 'targ': [], 'num': []})
        for src, targ in pairs:
            grouped = df[[src, targ, vals]].groupby([src, targ])[vals].sum().reset_index()
            grouped.columns = ['src', 'targ', 'num']
            stacked = pd.concat([stacked, grouped], axis=0)

    return stacked

def make_sankey(df, *cols, vals=None, **kwargs):
    """
    Create a Sankey diagram linking source values to target values with optional arguments.
    Args:
        df (DataFrame): Input DataFrame.
        cols (tuple): Source and target columns.
        vals (str): Column name for values.
        kwargs (dict): Additional customization options.
    """
    # Run stacking regardless of the number of columns
    df = stack_columns_to_dataframe(df, *cols, vals=vals)
    src, targ, vals = 'src', 'targ', 'num'
    # Modify the DataFrame
    df, labels = map_columns_to_numbers(df, src, targ)

    # Apply the threshold if provided
    val_min = kwargs.get('min_value', 0)
    if vals and val_min > 0:
        df = df[df[vals] >= val_min]

    # Assign values to links
    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    # Create the Sankey object
    link = {'source': df[src], 'target': df[targ], 'value': values}
    pad = kwargs.get('pad', 50)
    node = {'label': labels, 'pad': pad}
    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    # Customize the layout
    width = kwargs.get('width', 800)
    height = kwargs.get('height', 800)
    fig.update_layout(autosize=False, width=width, height=height)

    # Show the Sankey diagram
    fig.show()
