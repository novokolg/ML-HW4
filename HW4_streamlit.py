from sklearn.model_selection import train_test_split
# prophet
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics

import streamlit as st
import pandas as pd
import altair as alt



import joblib

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime as DT

st.title('Sales forecast accuracy (custom) for ML model based on fbprophet by city:')

#dataset source from gitlab
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv')
df['ds'] = df.date.apply(lambda x: DT.datetime.strptime(x, '%B-%Y').date())
df.rename(columns={'quantity':'y'}, inplace=True)
#df = df.set_index('ds')
df = df.sort_values(['city','ds'])

df = df.drop(['market', 'month', 'state', 'date', 'year'], axis = 1)
#df
#st.write("### Gross Agricultural Production ($B)", df.sort_index())

city_to_filter = st.selectbox("Choose cities", df.city.unique())
if not city_to_filter:
    st.error("Please select at city.")

def train_test_split(timeseries, test_size=12):
    return timeseries[:-test_size], timeseries[-test_size:]

df_city = df[df.city == city_to_filter]
train, test = train_test_split(df_city)

model = Prophet(interval_width=0.95,
                #     mcmc_samples=300,
                daily_seasonality=False,
                weekly_seasonality=False,
               ).fit(train)
#model.add_regressor('priceMin', prior_scale=0.5, mode='multiplicative')
#model.add_regressor('priceMax', prior_scale=0.5, mode='multiplicative')
#model.add_regressor('priceMod', prior_scale=0.5, mode='multiplicative')

future = model.make_future_dataframe(periods=12, freq='M')

forecast = model.predict(future)


# собственная метрика по которой сравниваем accuracy
def sales_forecasting_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    assert y_true.size == y_pred.size

    mask = y_true != 0
    if y_true[mask].size == 0:
        return np.nan

    return 1 - np.sum(np.abs(y_pred - y_true)[mask]) / np.sum(y_true[mask])

fig = plot_plotly(model, forecast, trend=True, changepoints=True, xlabel='date', ylabel='quantities', figsize=(1200,800))

# custom styles
fig.update_traces(mode='lines', selector=dict(name='Actual'))
fig.update_layout(title_text=f"SFA: {sales_forecasting_accuracy(test['y'], forecast['yhat'][-12:].round().astype(int).clip(lower=0))}")
fig.show()
