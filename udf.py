import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

## Data Input / Output and Conversion

## Data Exploration
# checking type of features within a dataframe
def func_df_col_names_by_type(df, flag_print = False):
    """function to return and print the data types of columns within a dataframe
    Keyword arguments: df (dataframe), flag_print (default = False); Return: cols_dict (dictionary), cols_df (dataframe)
    """

    # Identify binary cols
    numerical_cols = df.select_dtypes(include = ['number'], exclude = None).columns.tolist()
    # binary numerical cols (0, 1), need to check if all binary cols are coded as 0,1
    binary_cols = [col for col in df if df[col].dropna().value_counts().index.isin([0,1]).all()]
    # numerical cols (exclude binary)
    numerical_cols_exclude_binary = [e for e in numerical_cols if e not in binary_cols]
    # categorical cols (include binary)
    categorical_cols = df.select_dtypes(include = ['object'], exclude = None).columns.tolist()
    categorical_cols_include_binary = categorical_cols + binary_cols

    #  check through all data types using .select_dtypes()
    cols_dict = {
        "cols_number": numerical_cols,
        "cols_string": categorical_cols,
        "cols_binary": binary_cols,
        "cols_number_exclude_binary": numerical_cols_exclude_binary,
        "cols_string_include_binary": categorical_cols_include_binary,
        "cols_datetime": df.select_dtypes(include = ['datetime','datetime64'], exclude = None).columns.tolist(),
        "cols_timedelta": df.select_dtypes(include = ['timedelta','timedelta64'], exclude = None).columns.tolist(),
        "cols_category": df.select_dtypes(include = ['category'], exclude = None).columns.tolist(),
        "cols_others": df.select_dtypes(include = None,
                                            exclude = ['number','object','datetime','datetime64','timedelta','timedelta64','category']).columns.tolist()
    }

    # print all data type of all cols {col_type: col_name}
    if flag_print:
        for key,value in cols_dict.items():
            for v in value:
                print("{}:{}".format(key,v))

    # save to dataframe format
    feature_type = []
    feature_num = []
    feature_list = []
    for key,value in cols_dict.items():
        feature_type.append(key)
        feature_num.append(len(value))
        feature_list.append(value)
    cols_df = pd.DataFrame({'feature_type':feature_type, 'feature_num':feature_num, 'feature_list':feature_list})

    return cols_dict, cols_df


# function similar to describe() with missing value
def func_df_describe_all(df): ## input a dataframe
    """function similar to describe() with missing value
    Keyword arguments: df (dataframe); Return: df_summary
    """
    df_summary = df.describe(include='all').T
    df_summary['miss_perc'] = (df.isnull().sum()/df.shape[0]*100).values
    return df_summary


# function to extend the display of jupyter-notebook
def func_df_display_all(df,max_rows=1000,max_cols=1000):
    """function similar to display, but temporarily extend the max number of rows and columns
    Keyword arguments: df (dataframe), max_rows, max_cols; Return: display
    """
    import pandas as pd
    ### display more info by extending max_rows, max_columns
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_cols):
        display(df)


# function to examine the frequency and response level within a categorical variable
def func_cat_crosstab(df_X, y_binary, sortby='Frequency'):
    """function to examine the frequency and response level within a categorical variable
    Keyword arguments: df_X (a categorical feature), y_binary (label with 1/0), sortby='Frequency'/'Response_Rate'; Return: df_ct (crosstab dataframe), plot Frequency vs Response
    """

    df_ct = pd.crosstab(df_X, y_binary, margins=False)
    # output from pd.crosstab
    # y   0   1
    # x
    # 0  217  11
    # 1  158  14
    # ...
    # n  k0   k1

    df_ct['Count'] = df_ct.sum(axis=1)
    df_ct['Frequency'] = df_ct['Count']/df_ct['Count'].sum()*100
    df_ct['Frequency'] = df_ct['Frequency'].round(decimals=2)
    df_ct['Response_Rate'] =df_ct[1]/df_ct['Count']*100
    df_ct['Response_Rate'] = df_ct['Response_Rate'].round(decimals=2)
    df_ct = df_ct.sort_values(by=sortby,ascending=False)
    df_ct['Frequency_Cumu'] = df_ct['Frequency'].cumsum()
    plt.figure()
    plt.scatter(df_ct['Frequency'],df_ct['Response_Rate'])
    plt.xlabel('Frequency')
    plt.ylabel('Response Rate')
    plt.tight_layout()
    plt.show()
    return df_ct


## Feature Engineering
## DateTime
def func_add_datepart(df, fldname, drop=True, time=False, errors="raise", attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.

    Examples:
    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df

        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13

    >>> add_datepart(df, 'A')
    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    # attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9  # in seconds since 1970-01-02
    if drop: df.drop(fldname, axis=1, inplace=True)


## Correlation Analysis

def func_stat_cramers_v(x,y):
    import pandas as pd
    from scipy import stats
    confusion_matrix = pd.crosstab(index = x,
                                   columns = y,
                                   margins = True)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def func_df_cramers_v(dfx,dfy):
    cramers_v_vals = []
    for col in dfx.columns:
        cramers_v_val = func_stat_cramers_v(dfx[col],dfy)
        cramers_v_vals.append(cramers_v_val)
        #print("{}: {:.4f}".format(col,cramers_v_val))

    df_crmaers_v = pd.DataFrame({'feature': dfx.columns,'cramers_v': cramers_v_vals})
    df_crmaers_v.sort_values(by=['cramers_v'], ascending=False, inplace=True)
    return df_crmaers_v


## Visualizatoin ##########################################
# function to plot the normalized probability density distribution colored by label
def func_eda_hist_by_label_plot(df_X, Y, prefix='', figsize=(6,4), dir_png='../reports/figures/'):
    """function to plot the normalized probability density distribution
    Keyword arguments: df_X (feature dataframe), y (label pandas series), dir_png (output dir); Return: PNG files
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y_uniques = Y.unique()  # unique labels, e.g. 0 or 1

    for col in df_X.columns:
        # filter nan vales
        mask = ~df_X[col].isnull()
        x = df_X.loc[mask, col]
        y = Y[mask]

        # plot
        fig = plt.figure(figsize=figsize)
        plt.hist([x[y == y_unique] for y_unique in y_uniques],
                 label=y_uniques,
                 weights=[np.ones(x[y == y_unique].count()) / x[y == y_unique].count() for y_unique in y_uniques])#density=True)
        plt.xlabel(col)
        plt.ylabel('Normalized Probability Distribution')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return
