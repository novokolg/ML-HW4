from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_cross_validation_metric

import streamlit as st
import base64
import numpy as np
import pandas as pd
import datetime as DT

@st.cache
def data_loading(filename, dlm, sp):
    df = pd.read_csv(filename, sep = sp, delimiter = dlm, parse_dates = True)
    return df

def use_own_dataset(df):
    st.write("Data for modeling:")
    st.write(df)
    if len(df.columns) != 2:
        st.write('Error: only 2 columns were expected')
        st.stop()
    date_column = st.selectbox("Choose column with date ", df.columns)
    if df.columns[0] == date_column:
        target_column = df.columns[1]
    else:
        target_column = df.columns[0]
        target_column = df.columns[0]

#    df[date_column] = df[date_column].apply(lambda x: DT.datetime.strptime(x,'%Y-%m-%d').date())

    df.rename(columns={date_column: 'ds'}, inplace=True)
    df.rename(columns={target_column: 'y'}, inplace=True)
    return df

def use_default_dataset():
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv')
    df['ds'] = df.date.apply(lambda x: DT.datetime.strptime(x, '%B-%Y').date())
    city_to_filter = "MUMBAI"
    df.rename(columns={'quantity': 'y'}, inplace=True)
    # df = df.set_index('ds')
    df = df.sort_values(['city', 'ds'])
    city_to_filter = "MUMBAI"
    df = df[df.city == city_to_filter]
    df = df.drop(['market', 'month', 'state', 'date', 'year', 'city', 'priceMin', 'priceMax', 'priceMod'], axis=1)
    st.write("Data for modeling:")
    st.write(df)
    return df


def train_test_split(timeseries, test_size):
    return timeseries[:-test_size], timeseries[-test_size:]


def sales_forecasting_accuracy(y_true, y_pred):
    '''собственная метрика по которой сравниваем accuracy'''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert y_true.size == y_pred.size
    mask = y_true != 0
    if y_true[mask].size == 0:
        return np.nan
    return 1 - np.sum(np.abs(y_pred - y_true)[mask]) / np.sum(y_true[mask])

def data_granularity_processing(option2):
    if option2 == 'Yearly':
        period_freq = 'YS'
        daily_season = False
        weekly_season = False
    if option2 == 'Monthly':
        period_freq = 'MS'
        daily_season = False
        weekly_season = False
    if option2 == 'Weekly':
        period_freq = 'W'
        daily_season = False
        weekly_season = True
    if option2 == 'Daily':
        period_freq = 'D'
        daily_season = True
        weekly_season = True

    return period_freq, daily_season, weekly_season


def forecasting(data, forecasting_periods, period_freq, daily_season, weekly_season):
#    if period_freq == 'W':
#        data.ds = data.ds.apply(lambda x: x - DT.timedelta(days = x.weekday()))
    train, test = train_test_split(data, forecasting_periods)
    model = Prophet(daily_seasonality = daily_season, weekly_seasonality = weekly_season).fit(train)
    #    interval_width = 0.95,
    # validation on last periods = forecasting_periods
    future = model.make_future_dataframe(periods=forecasting_periods, freq=period_freq)
    forecast = model.predict(future)
    val = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(data.set_index('ds'))
    val['e'] = val['y'] - val['yhat']
    val['p'] = 100 * val['e'] / val['y']
    accuracy_mape = "MAPE, %: " + str(np.mean(abs(val[-forecasting_periods:]['p'])))

    model = Prophet(daily_seasonality = daily_season, weekly_seasonality = weekly_season).fit(data)
    future = model.make_future_dataframe(periods=forecasting_periods,  freq = period_freq)
#    st.write(future)
    forecast = model.predict(future)
    st.write("Forecast for future periods:")

    forecast.rename(columns={'yhat': 'Predictions'}, inplace=True)
    st.write(forecast.set_index('ds')[['Predictions']][-forecasting_periods:])

    st.write('Make a chart with forecasted values and data validation')
    val = forecast.set_index('ds')[['Predictions']].join(data.set_index('ds'))
#    val.rename(columns={'yhat': 'Predictions'}, inplace=True)
    val.rename(columns={'y': 'Actuals'}, inplace=True)
#    fig = plot_plotly(model, forecast, trend=True, changepoints=True, xlabel='date', ylabel='quantities', figsize=(1200,800))
                # custom styles
#    fig.update_traces(mode='lines', selector=dict(name='Actual'))
#    fig.update_layout(title_text=accuracy_mape)
#    fig.show()

    chart_data = pd.DataFrame(val, columns=['Predictions', 'Actuals'])
    st.line_chart(chart_data)

def main():
    st.title('This tool helps you to make brief forecast for time series using fbprophet:')
    option = st.selectbox('Do you like to use your own dataset or default dataset?',
                          ('', 'Own dataset', 'Use default dataset as an example'))
    st.write('You selected:', option)

    if option == 'Own dataset':
        filename = st.file_uploader("Choose file for loading. \n Please, note that the CSV file should contain exactly ",
                                    folder="my_folder", type=("csv", "txt"))
        if filename is not None:
            dlm = st.sidebar.text_input('Set up delimeter:', ';')
            sp = st.sidebar.text_input('Set up separator:', '')
            # date_format = st.text_input('Set up date format:', '%Y-%m-%d')
            data = data_loading(filename, dlm, sp)
            use_own_dataset(data)
        else:
            st.text("Please, provide data for forecasting.")
            st.stop()
    elif option == 'Use default dataset as an example':
        data = use_default_dataset()
    else:
        st.text("Please, provide data for forecasting.")
        st.stop()

    if data is not None:
#        forecasting_periods, period_freq, daily_season, weekly_season = preparing_for_forecasting()
        st.sidebar.title('Please choose a granularity of your data:')
        option2 = st.sidebar.selectbox('',('Yearly', 'Monthly', 'Weekly', 'Daily'))
        period_freq, daily_season, weekly_season = data_granularity_processing(option2)

        forecasting_periods = st.sidebar.slider('How many datapoint you want to forecast:', 1, 365)

        forecasting(data, forecasting_periods, period_freq, daily_season, weekly_season)
    else:
        st.write('Error: Data is None')




if __name__ == "__main__":
    main()