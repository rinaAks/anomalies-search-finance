# Описание наборов данных

**Биржа NYSE (сектор здравоохранения - Healthcare).** 

**Выборка 1:** Признаки EPS Growth (TTM*), Sales Growth (TTM*), **Выборка 2:** Признаки Beta (1y), P/E Ratio. 

Компании, чья капитализация составляет больше 1 млрд. долларов США

**Trailing Twelve Months (TTM)*


```python
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

https://finance.yahoo.com/research-hub/screener/most_actives/?start=0&count=100

Применила фильтры:

Exchange: NYSE, Sector: Healthcare, Market Cap (intraday): больше 1B


```python
# список коротких названий компаний
ticker_list = ['PFE', 'HLN', 'MDT', 'CVS', 'MRK', 
               'TEVA', 'AVTR', 'BMY', 'ABT', 'AGL',
               'WRBY', 'BSX', 'TDOC', 'NVO', 'JNJ',
               'ABBV', 'NVS', 'OSCR', 'EW', 'GSK']
```

Среди полей не было P/E Ratio, наиболее близкое trailingPE

Trailing P/E Ratio is calculated by dividing a company's current share price by its most recent reported earnings per share (EPS)


```python
# сохраняем нужные столбцы
data = []
for ticker in ticker_list:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    eps_growth = info.get('earningsGrowth', np.nan)   # EPS Growth (TTM)
    sales_growth = info.get('revenueGrowth', np.nan)  # Sales Growth (TTM)
    beta = info.get('beta', np.nan)                   # Beta (1y)
    pe_ratio = info.get('trailingPE', np.nan)         # P/E Ratio
    market_cap = info.get('marketCap', 0) / 1e9       # Капитализация
    
    data.append({
        'Company': ticker,
        'EPS Growth (TTM)': eps_growth,
        'Sales Growth (TTM)': sales_growth,
        'Beta (1y)': beta,
        'P/E Ratio': pe_ratio,
        'MarketCap (B)': market_cap
    })
    
    print(ticker, " - ", info.get('shortName', 'N/A'))
```

    PFE  -  Pfizer, Inc.
    HLN  -  Haleon plc
    MDT  -  Medtronic plc.
    CVS  -  CVS Health Corporation
    MRK  -  Merck & Company, Inc.
    TEVA  -  Teva Pharmaceutical Industries 
    AVTR  -  Avantor, Inc.
    BMY  -  Bristol-Myers Squibb Company
    ABT  -  Abbott Laboratories
    AGL  -  agilon health, inc.
    WRBY  -  Warby Parker Inc.
    BSX  -  Boston Scientific Corporation
    TDOC  -  Teladoc Health, Inc.
    NVO  -  Novo Nordisk A/S
    JNJ  -  Johnson & Johnson
    ABBV  -  AbbVie Inc.
    NVS  -  Novartis AG
    OSCR  -  Oscar Health, Inc.
    EW  -  Edwards Lifesciences Corporatio
    GSK  -  GSK plc
    


```python
# Этот код выводит все trailingPE, можно использовать для сравнения с тем, что есть на сайте
'''
for ticker in ticker_list:
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # PE = stock's share price / the earnings per share (EPS)
    print(ticker, " - ", info.get('trailingPE', np.nan))
'''
```




    '\nfor ticker in ticker_list:\n    stock = yf.Ticker(ticker)\n    info = stock.info\n    \n    # PE = stock\'s share price / the earnings per share (EPS)\n    print(ticker, " - ", info.get(\'trailingPE\', np.nan))\n'




```python
# Этот код выводит всю информацию компании TEVA
'''
stock = yf.Ticker(ticker_list[4])
info = stock.info
print(info)
'''
```




    '\nstock = yf.Ticker(ticker_list[4])\ninfo = stock.info\nprint(info)\n'




```python
df = pd.DataFrame(data)
```


```python
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>EPS Growth (TTM)</th>
      <th>Sales Growth (TTM)</th>
      <th>Beta (1y)</th>
      <th>P/E Ratio</th>
      <th>MarketCap (B)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PFE</td>
      <td>NaN</td>
      <td>0.219</td>
      <td>0.543</td>
      <td>17.404257</td>
      <td>139.177394</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HLN</td>
      <td>NaN</td>
      <td>-0.003</td>
      <td>0.227</td>
      <td>24.829270</td>
      <td>45.984891</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MDT</td>
      <td>0.019</td>
      <td>0.025</td>
      <td>0.823</td>
      <td>26.978659</td>
      <td>113.491968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CVS</td>
      <td>-0.178</td>
      <td>0.036</td>
      <td>0.531</td>
      <td>18.573770</td>
      <td>85.709185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MRK</td>
      <td>NaN</td>
      <td>0.068</td>
      <td>0.363</td>
      <td>12.925817</td>
      <td>220.068610</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TEVA</td>
      <td>NaN</td>
      <td>-0.051</td>
      <td>0.826</td>
      <td>NaN</td>
      <td>17.519895</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AVTR</td>
      <td>4.208</td>
      <td>-0.021</td>
      <td>1.301</td>
      <td>15.192308</td>
      <td>10.766468</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BMY</td>
      <td>-0.959</td>
      <td>0.075</td>
      <td>0.436</td>
      <td>NaN</td>
      <td>121.169953</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ABT</td>
      <td>4.794</td>
      <td>0.072</td>
      <td>0.693</td>
      <td>17.236912</td>
      <td>228.392600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AGL</td>
      <td>NaN</td>
      <td>0.442</td>
      <td>0.677</td>
      <td>NaN</td>
      <td>1.682196</td>
    </tr>
    <tr>
      <th>10</th>
      <td>WRBY</td>
      <td>NaN</td>
      <td>0.178</td>
      <td>1.921</td>
      <td>NaN</td>
      <td>2.228640</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BSX</td>
      <td>0.120</td>
      <td>0.224</td>
      <td>0.774</td>
      <td>80.992000</td>
      <td>149.741044</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TDOC</td>
      <td>NaN</td>
      <td>-0.030</td>
      <td>1.272</td>
      <td>NaN</td>
      <td>1.347695</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NVO</td>
      <td>0.292</td>
      <td>0.301</td>
      <td>0.167</td>
      <td>20.677810</td>
      <td>308.316701</td>
    </tr>
    <tr>
      <th>14</th>
      <td>JNJ</td>
      <td>-0.034</td>
      <td>0.053</td>
      <td>0.463</td>
      <td>26.468048</td>
      <td>369.309516</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ABBV</td>
      <td>NaN</td>
      <td>0.056</td>
      <td>0.598</td>
      <td>86.305435</td>
      <td>364.887507</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NVS</td>
      <td>-0.656</td>
      <td>0.151</td>
      <td>0.490</td>
      <td>18.641157</td>
      <td>219.908342</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OSCR</td>
      <td>NaN</td>
      <td>0.671</td>
      <td>1.752</td>
      <td>129.700000</td>
      <td>3.249867</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EW</td>
      <td>0.071</td>
      <td>0.094</td>
      <td>1.123</td>
      <td>30.713678</td>
      <td>42.098213</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GSK</td>
      <td>0.173</td>
      <td>0.008</td>
      <td>0.286</td>
      <td>23.521738</td>
      <td>76.960170</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20 entries, 0 to 19
    Data columns (total 6 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   Company             20 non-null     object 
     1   EPS Growth (TTM)    11 non-null     float64
     2   Sales Growth (TTM)  20 non-null     float64
     3   Beta (1y)           20 non-null     float64
     4   P/E Ratio           15 non-null     float64
     5   MarketCap (B)       20 non-null     float64
    dtypes: float64(5), object(1)
    memory usage: 1.1+ KB
    

По параметрам P/E Ratio и EPS Growth есть много пропусков. Заполним недостающие значения данными с сайта ivesting.com. Заполним те, где был только 1 пропуск


```python
df.at[0, 'EPS Growth (TTM)'] = 0.2776  # PFE
df.at[4, 'EPS Growth (TTM)'] = 4.5989 # MRK
# было ещё несколько с одним пропуском, но среди топ 300 на investing их не было..
```


```python
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>EPS Growth (TTM)</th>
      <th>Sales Growth (TTM)</th>
      <th>Beta (1y)</th>
      <th>P/E Ratio</th>
      <th>MarketCap (B)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PFE</td>
      <td>0.2776</td>
      <td>0.219</td>
      <td>0.543</td>
      <td>17.404257</td>
      <td>139.177394</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HLN</td>
      <td>NaN</td>
      <td>-0.003</td>
      <td>0.227</td>
      <td>24.829270</td>
      <td>45.984891</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MDT</td>
      <td>0.0190</td>
      <td>0.025</td>
      <td>0.823</td>
      <td>26.978659</td>
      <td>113.491968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CVS</td>
      <td>-0.1780</td>
      <td>0.036</td>
      <td>0.531</td>
      <td>18.573770</td>
      <td>85.709185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MRK</td>
      <td>4.5989</td>
      <td>0.068</td>
      <td>0.363</td>
      <td>12.925817</td>
      <td>220.068610</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TEVA</td>
      <td>NaN</td>
      <td>-0.051</td>
      <td>0.826</td>
      <td>NaN</td>
      <td>17.519895</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AVTR</td>
      <td>4.2080</td>
      <td>-0.021</td>
      <td>1.301</td>
      <td>15.192308</td>
      <td>10.766468</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BMY</td>
      <td>-0.9590</td>
      <td>0.075</td>
      <td>0.436</td>
      <td>NaN</td>
      <td>121.169953</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ABT</td>
      <td>4.7940</td>
      <td>0.072</td>
      <td>0.693</td>
      <td>17.236912</td>
      <td>228.392600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AGL</td>
      <td>NaN</td>
      <td>0.442</td>
      <td>0.677</td>
      <td>NaN</td>
      <td>1.682196</td>
    </tr>
    <tr>
      <th>10</th>
      <td>WRBY</td>
      <td>NaN</td>
      <td>0.178</td>
      <td>1.921</td>
      <td>NaN</td>
      <td>2.228640</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BSX</td>
      <td>0.1200</td>
      <td>0.224</td>
      <td>0.774</td>
      <td>80.992000</td>
      <td>149.741044</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TDOC</td>
      <td>NaN</td>
      <td>-0.030</td>
      <td>1.272</td>
      <td>NaN</td>
      <td>1.347695</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NVO</td>
      <td>0.2920</td>
      <td>0.301</td>
      <td>0.167</td>
      <td>20.677810</td>
      <td>308.316701</td>
    </tr>
    <tr>
      <th>14</th>
      <td>JNJ</td>
      <td>-0.0340</td>
      <td>0.053</td>
      <td>0.463</td>
      <td>26.468048</td>
      <td>369.309516</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ABBV</td>
      <td>NaN</td>
      <td>0.056</td>
      <td>0.598</td>
      <td>86.305435</td>
      <td>364.887507</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NVS</td>
      <td>-0.6560</td>
      <td>0.151</td>
      <td>0.490</td>
      <td>18.641157</td>
      <td>219.908342</td>
    </tr>
    <tr>
      <th>17</th>
      <td>OSCR</td>
      <td>NaN</td>
      <td>0.671</td>
      <td>1.752</td>
      <td>129.700000</td>
      <td>3.249867</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EW</td>
      <td>0.0710</td>
      <td>0.094</td>
      <td>1.123</td>
      <td>30.713678</td>
      <td>42.098213</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GSK</td>
      <td>0.1730</td>
      <td>0.008</td>
      <td>0.286</td>
      <td>23.521738</td>
      <td>76.960170</td>
    </tr>
  </tbody>
</table>
</div>



Остальные выкидываем


```python
df = df.dropna()
```


```python
df.count()
```




    Company               12
    EPS Growth (TTM)      12
    Sales Growth (TTM)    12
    Beta (1y)             12
    P/E Ratio             12
    MarketCap (B)         12
    dtype: int64



Разделим на две выборки


```python
# 1: Признаки EPS Growth (TTM), Sales Growth (TTM)
# 2: Признаки Beta (1y), P/E Ratio
df1 = df[['Company', 'EPS Growth (TTM)', 'Sales Growth (TTM)']]
df2 = df[['Company', 'Beta (1y)', 'P/E Ratio']]
```

# Выявление аномалий

Методы:
1. Одноклассовый метод опорных векторов (SVM)
2. Эллиптическая оболочка (Elliptic Envelope)
3. Фактор локального выброса (Local outlier factor)
4. Изолированный лес (Isolation forest)


```python
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
```

## Параметры 1

EPS Growth (TTM), Sales Growth (TTM)

### Параметры оценщиков, использованные для анализа


```python
features = ['EPS Growth (TTM)', 'Sales Growth (TTM)']
X = df1[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

LocalOutlierFactor без параметра contamination плохо работал, на остальных методах значения параметров как будто не влияли. Возможно это из-за малого количества данных.


```python
estimators = {
    'One-Class SVM': OneClassSVM(nu=0.1),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.1),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=11, contamination=0.1),
    'Isolation Forest': IsolationForest(contamination=0.1)
}

results = {}
for name, estimator in estimators.items():
    if name == 'Local Outlier Factor':
        results[name] = estimator.fit_predict(X_scaled)
    else:
        results[name] = estimator.fit(X_scaled).predict(X_scaled)
```

### Диаграммы разброса


```python
plt.figure(figsize=(15, 10))
for i, (name, pred) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(X[:, 0], X[:, 1], c=pred)
    plt.title("Выборка 1, " + name)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
```


    
![png](anomalies-search-finance_files/anomalies-search-finance_29_0.png)
    


### Ансамблирование и подсчёт аномалий


```python
ensemble_votes = np.array([results[name] for name in estimators.keys()]).T
ensemble_result = []
for row in ensemble_votes:
    outlier_count = np.sum(row == -1)
    if outlier_count >= 3:
        ensemble_result.append(3)  # Аномалия
    elif outlier_count == 2:
        ensemble_result.append(2)   # Подозрение 2
    elif outlier_count == 1:
        ensemble_result.append(1)   # Подозрение 1
    else:
        ensemble_result.append(0)   # Нормальные данные
```


```python
df1['Ensemble'] = ensemble_result
anomalies = df1[df1['Ensemble'] == 3]['Company'].tolist()
suspicious2 = df1[df1['Ensemble'] == 2]['Company'].tolist()
suspicious1 = df1[df1['Ensemble'] == 1]['Company'].tolist()
print("Аномалии : ", anomalies)        # все выбрали как аномальное
print("Подозрительные для двоих: ", suspicious2)  # аномальным считают двое
print("Подозрительные для одного: ", suspicious1)  # аномальным считает хотя бы один
```

    Аномалии :  []
    Подозрительные для двоих:  ['NVO']
    Подозрительные для одного:  ['MRK', 'AVTR', 'ABT', 'JNJ', 'EW', 'GSK']
    

    C:\Users\Asus\AppData\Local\Temp\ipykernel_5332\2312458526.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['Ensemble'] = ensemble_result
    


```python
# Визуализация ансамбля
plt.figure(figsize=(5, 3))
plt.scatter(X[:, 0], X[:, 1], c = ensemble_result)
plt.title("Выборка 1, ансамбль")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.tight_layout()
plt.show()
```


    
![png](anomalies-search-finance_files/anomalies-search-finance_33_0.png)
    


## Параметры 2

Beta (1y), P/E Ratio

### Параметры оценщиков, использованные для анализа


```python
features = ['P/E Ratio','Beta (1y)']
X = df2[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```


```python
estimators = {
    'One-Class SVM': OneClassSVM(nu=0.1),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.1),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=11, contamination=0.1),
    'Isolation Forest': IsolationForest(contamination=0.1)
}

results = {}
for name, estimator in estimators.items():
    if name == 'Local Outlier Factor':
        results[name] = estimator.fit_predict(X_scaled)
    else:
        results[name] = estimator.fit(X_scaled).predict(X_scaled)
```

### Диаграммы разброса


```python
plt.figure(figsize=(15, 10))
for i, (name, pred) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(X[:, 0], X[:, 1], c=pred)
    plt.title("Выборка 2, " + name)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
```


    
![png](anomalies-search-finance_files/anomalies-search-finance_39_0.png)
    


Интересно, что Local outlier factor не сочёл за аномальную правую одинокую точку. 

### Ансамблирование и подсчёт аномалий


```python
ensemble_votes = np.array([results[name] for name in estimators.keys()]).T
ensemble_result = []
for row in ensemble_votes:
    outlier_count = np.sum(row == -1)
    if outlier_count >= 3:
        ensemble_result.append(3)  # Аномалия
    elif outlier_count == 2:
        ensemble_result.append(2)   # Подозрение 2
    elif outlier_count == 1:
        ensemble_result.append(1)   # Подозрение 1
    else:
        ensemble_result.append(0)   # Нормальные данные
```


```python
df1['Ensemble'] = ensemble_result
anomalies = df1[df1['Ensemble'] == 3]['Company'].tolist()
suspicious2 = df1[df1['Ensemble'] == 2]['Company'].tolist()
suspicious1 = df1[df1['Ensemble'] == 1]['Company'].tolist()
print("Аномалии : ", anomalies)        # все выбрали как аномальное
print("Подозрительные для двоих: ", suspicious2)  # аномальным считают двое
print("Подозрительные для одного: ", suspicious1)  # аномальным считает хотя бы один
```

    Аномалии :  ['AVTR', 'BSX']
    Подозрительные для двоих:  []
    Подозрительные для одного:  ['MDT', 'ABT', 'NVO', 'EW']
    

    C:\Users\Asus\AppData\Local\Temp\ipykernel_5332\2312458526.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['Ensemble'] = ensemble_result
    


```python
# Визуализация ансамбля
plt.figure(figsize=(5, 3))
plt.scatter(X[:, 0], X[:, 1], c = ensemble_result)
plt.title("Выборка 2, ансамбль")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.tight_layout()
plt.show()
```


    
![png](anomalies-search-finance_files/anomalies-search-finance_44_0.png)
    


# Выводы

**Результаты по параметрам EPS Growth (TTM), Sales Growth (TTM)**

* Аномалии :  []
* Подозрительные для двоих:  ['NVO']
* Подозрительные для одного:  ['MRK', 'AVTR', 'ABT', 'JNJ', 'EW', 'GSK']

**Результаты по параметрам P/E Ratio, Beta (1y)**

* Аномалии :  ['AVTR', 'BSX']
* Подозрительные для двоих:  []
* Подозрительные для одного:  ['MDT', 'ABT', 'NVO', 'EW']

По обеим группам параметров в выбросы/подозрения попали: AVTR, ABT, NVO, EW.
