import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import calendar


marketing_data = pd.read_csv('MarketingSpend.csv', header=0, names=['Date', 'Offline', 'Online'])
retail_data = pd.read_csv('Retail.csv')


def analyze_marketing(df: pd.DataFrame):
    stats_summary = {
        'Online': {
            'mean': round(df['Online'].mean(), 3),
            'max': df['Online'].max(),
            'min': df['Online'].min(),
            'std': round(df['Online'].std(), 3),
            'var': round(df['Online'].var(), 3)
        },
        'Offline': {
            'mean': round(df['Offline'].mean(), 3),
            'max': df['Offline'].max(),
            'min': df['Offline'].min(),
            'std': round(df['Offline'].std(), 3),
            'var': round(df['Offline'].var(), 3)
        }
    }

    for mode in stats_summary:
        print(f"{mode} среднее значение: {stats_summary[mode]['mean']}")
        print(f"{mode} максимум: {stats_summary[mode]['max']}")
        print(f"{mode} минимум: {stats_summary[mode]['min']}")
        print(f"{mode} стандарт. отклонение: {stats_summary[mode]['std']}")
        print(f"{mode} дисперсия: {stats_summary[mode]['var']}")



def draw_distribution(data_column: str, ax, shift=1e-5):
    values = sorted(marketing_data[data_column])
    mu, sigma = norm.fit(values)
    density_curve = norm.pdf(values, mu, sigma)

    quartiles = marketing_data[data_column].quantile([.25, .5, .75])
    max_y = [density_curve[values.index(q)] if q in values else 0 for q in quartiles]
    ax.vlines(x=quartiles, ymin=0, ymax=max_y, linestyle='--', color='gray')

    ax.plot(values, density_curve, color='teal')
    ax.set_ylim([0, max(density_curve) + shift])
    ax.set_title(f'Распределение: {data_column}')
    ax.legend(['Плотность', 'Квартили'])


def display_distributions():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    draw_distribution('Online', axs[0])
    draw_distribution('Offline', axs[1])
    fig.suptitle('График нормального распределения')
    plt.tight_layout()
    plt.show()



def summarize_retail():
    total_entries = retail_data['InvoiceNo'].count()
    unique_orders = retail_data['InvoiceNo'].nunique()

    print(f"Всего записей: {total_entries}")
    print(f"Уникальных заказов: {unique_orders}")

    top_codes = (retail_data
                 .groupby('StockCode')['Quantity']
                 .sum()
                 .sort_values(ascending=False)
                 .head(10))
    print("\nНаиболее часто заказываемые товары:")
    print(top_codes)



def generate_visuals():
    marketing_cp = marketing_data.copy()
    marketing_cp['Date'] = pd.to_datetime(marketing_cp['Date'])
    marketing_cp.set_index('Date', inplace=True)
    monthly_stats = marketing_cp.resample('M').sum()
    monthly_stats.index = monthly_stats.index.month
    monthly_stats.index = monthly_stats.index.map(lambda x: calendar.month_abbr[x])
    monthly_stats[['Offline', 'Online']].plot(kind='barh', stacked=True, colormap='coolwarm')
    plt.xlabel('Продажи')
    plt.ylabel('Месяц')
    plt.title('Суммарные продажи по месяцам')
    plt.tight_layout()
    plt.show()

    retail_cp = retail_data.copy()
    retail_cp['InvoiceDate'] = pd.to_datetime(retail_cp['InvoiceDate'])
    daily_totals = (retail_cp
                    .groupby(retail_cp['InvoiceDate'].dt.date)['Quantity']
                    .sum()
                    .reset_index())
    daily_totals['DayOfYear'] = pd.to_datetime(daily_totals['InvoiceDate']).dt.dayofyear
    plt.scatter(daily_totals['DayOfYear'], daily_totals['Quantity'], alpha=0.7,
                s=daily_totals['Quantity'] / 10, c='darkorange')
    plt.title('Объём заказов по дням года')
    plt.xlabel('День года')
    plt.ylabel('Общее количество')
    plt.grid(True, linestyle=':')
    plt.show()



analyze_marketing(marketing_data)
display_distributions()
summarize_retail()
generate_visuals()
