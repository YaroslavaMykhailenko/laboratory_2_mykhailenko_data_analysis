import pandas as pd
import numpy as np
from statistics import pvariance
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import scipy.stats as st
import numpy as np
import pandas as pd
from math import sqrt
from plotly.subplots import make_subplots
from prettytable import PrettyTable
import plotly.graph_objects as go
from numpy import arange, cos, pi, insert, append, sin, sqrt



# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

import_data = pd.read_csv('A12.txt', header=None)
import_data.columns = [f'{i}' for i in range(12)]
print(import_data)

# table_format = PrettyTable()
# table_format.add_column('Фактор', ['Parametr'])
# for column_name, column_data in import_data.iteritems():

#     table_format.add_column(f'A{str(int(column_name)+1)}', [round(column_data, 2).to_string(index=False)])

# # print(table_format)


# середнє і дисперсія
for column_name, column_data in import_data.iteritems():
    print('➊------------------➊')
    print(f' Заданий параметр:{int(column_name) + 1}')
    print('➊------------------➊')
    mean_value = column_data.mean()
    dispersion_value = column_data.var()
    print("• Середнє знач параметру:", round(mean_value, 3))
    print("• Дисперсія параметру:", round(dispersion_value, 3))
    
# нормалізуємо 

import_data = (import_data - import_data.mean(axis=0)) / (import_data.std(axis=0))
# print(import_data)

# Обчислення кореляційної матриці
correlation_matrix = import_data.corr()
# print(correlation_matrix)


# Аналіз кореляційної матриці – виділити групу із трьох (чотирьох) параметрів, парна кореляція між якими велика (коефіцієнт кореляції близька по модулю до 1)

pair_coef_corr = correlation_matrix.unstack().sort_values() 

influent_coef_corr = pair_coef_corr[abs(pair_coef_corr) > 0.9]

coef_list = []
for i in range(0, influent_coef_corr.shape[0]-12, 2):
    coef_list.append(influent_coef_corr.index[i])

print(influent_coef_corr[coef_list])



# часткові коефіцієнти
ab, ac, ad, bc, bd, cd =  influent_coef_corr['7' , '8'], influent_coef_corr['7' , '9'],  influent_coef_corr['7' , '10'],\
    influent_coef_corr['8' , '9'], influent_coef_corr['8' , '10'], influent_coef_corr['9' , '10']

print(ab, ac, ad, bc, bd, cd)

r_ab_c = (ab - ac * bc) / (((1 - ac**2) * (1 - bc**2))**0.5)

print("\nЧастковий коефіцієнт кореляції між ознаками a та b без урахування впливу ознаки c: ", r_ab_c)


r_ac_b = (ac - ab * bc) / (((1 - ab**2) * (1 - bc**2))**0.5) 
print("\nЧастковий коефіцієнт кореляції між ознаками a та c без урахування впливу ознаки b: ", r_ac_b)

# проміжні дані
r_ad_c = (ad - ac * cd) / (((1 - ac**2) * (1 - cd**2))**0.5)
r_bd_c = (bd - bc * cd) / (((1 - bc**2) * (1 - cd**2))**0.5)
r_ad_b = (ad - ab * bd) / (((1 - ab**2) * (1 - bd**2))**0.5)
r_cd_b = (cd - bc * bd) / (((1 - bc**2) * (1 - bd**2))**0.5)
r_ab_cd = (r_ab_c - r_ad_c * r_bd_c) / (((1 - r_ad_c**2) * (1 - r_bd_c**2))**0.5)
r_ac_bd = (r_ac_b - r_ad_b * r_cd_b) / (((1 - r_ad_b**2) * (1 - r_cd_b**2))**0.5)
r_ad_bc = (r_ad_b - r_ac_b * r_cd_b)/(((1 - r_ac_b**2) * (1 - r_cd_b**2))**0.5) 

print("\nЧастковий коефіцієнт кореляції між ознаками a та b без урахування впливу ознаки c: ", r_ab_c)
print("\nЧастковий коефіцієнт кореляції між ознаками a та c без урахування впливу ознаки b: ", r_ac_b)
print("\nЧастковий коефіцієнт кореляції між ознаками a та b без урахування впливу ознаки c та d: ", r_ab_cd)
print("\nЧастковий коефіцієнт кореляції між ознаками a та с без урахування впливу ознаки b та d: ", r_ac_bd)
print("\nЧастковий коефіцієнт кореляції між ознаками a та d без урахування впливу ознаки b та c: ", r_ad_bc)

# знаходження множинного коефіцієнта кореляції для параметра а(лінійний двофакторний зв’язок)
r_a_bc = (((ab**2) + (ac**2) - 2 * ab * ac * bc) / (1 - bc**2))**0.5 

print("\nМножинний коефіцієнт кореляції для параметра a при лінійному двофакторному зв’язку з параметрами b та c", r_a_bc)
# Знайти множинний коефіцієнт кореляції для параметра a (лінійний трифакторний зв’язок)
R_a_bcd = 1 - (1 - ab**2) * (1 - r_ac_b**2) * (1 - r_ad_bc**2) 

print("\nМножинний коефіцієнт кореляції для параметра a при лінійному трифакторному зв’язку з параметрами b, c та d: ", R_a_bcd)


# Знаходження власних чисел матриці r з рівняння

num = np.linalg.eigh(correlation_matrix, UPLO="L")[0]
percentage_dispetion = num.copy()

for i in range(1,12):
    percentage_dispetion[10 - i] = percentage_dispetion[10 - i] + percentage_dispetion[10 - i + 1]
summary_dispersion = list(map(lambda x: 100 * x / percentage_dispetion[0], percentage_dispetion))

dataframe_ = pd.DataFrame(data={'Власні числа': num,'Частка дисперсії': percentage_dispetion, 'Сумарна дисперсія': summary_dispersion})

print(dataframe_)

# Графік для критерію кам’янистого осипу

plt.plot(list(range(1,13)), num[::-1], color='pink', marker='o')
plt.title('Графік для критерію кам’янистого осипу')
plt.grid()
plt.show()
# Критерій інформаційності 

Info_criterion = round(sum(num[9:]) / sum(num), 4)
print("Критерій інформаційності Ik =  ",Info_criterion)

# Обчислення власних векторів 

vector = np.linalg.eigh(correlation_matrix, UPLO="L")[1]
vectors = pd.DataFrame(vector)
print(vectors)

# Власний вектор максимального власного числа

print("Власний вектор максимального власного числа:",vectors[11])

# Перевірити виконання умов
for j in range(12):
    for k in range(12):
        if 0.99 < np.vstack(np.array(vectors[j])).T.dot(np.vstack(np.array(vectors[k]))) < 1.01:
            print(f"a'[{j}] * a[{k}] == 1")
        elif -10**(-10) < np.vstack(np.array(vectors[j])).T.dot(np.vstack(np.array(vectors[k]))) < 10**(-10):
            print(f"a'[{j}] * a[{k}] == 0")


# Знаходження головних компонентів
quantity_of_compon = 3
main_compon = pd.DataFrame()
for (columnName, columnData) in import_data.iteritems():
    ak = np.array(vectors[int(columnName)])
main_compon = import_data.mul(ak).dropna(how='all')
main_compon = main_compon[main_compon.columns[-(quantity_of_compon):]]

# Перші три головні фактори
print('\nПерші три головні фактори: \n')
print(main_compon)

# Їх Графіки
_, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
ax = axs.flatten()

for column_name, column_data in main_compon.iteritems():

    ax[int(int(column_name)%3)].plot(column_data, color='indigo')
    ax[int(int(column_name)%3)].set(ylabel=f'A{int(column_name) + 1}')

plt.tight_layout()
plt.show()


# # Графік сигналу для Ν=420
uf = scipy.stats.uniform()

n = 12-15
N = 420
M = 12 # згідно з Варінтом 12
ng = 92  # згідно з номером групи
T= [0,1] # Сигнал задано на проміжку

#  Графік сигналу для Ν=420  та Ν=4096 
i = np.arange(N)

for N in (420, 4096): # s2 remembers last

    s2 = 2 * uf.rvs(1) + ng * cos(2 * M * pi * i / N) * (1 + 0.1 * uf.rvs(1)) + 17 * cos(
        4 * M * pi * i / N + uf.rvs(1)) + 3 * cos(5 * M * pi * i / N) * (uf.rvs(1) + ng)
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.plot(s2, color='crimson')
    plt.show()


# проведемо нормування
s_mean = (1/N)* sum(s2)
s_std_squared = (1 / (N - 1) ) * sum(pow(s2 - s_mean, 2)) 
s2 = (s2 - s_mean)/s_std_squared
print('s_mean', s_mean)
print('s_std_squared', s_std_squared)
print('S2:', s2)


def transformation(s):

    a = np.array([(2 / N) * sum(s * cos(2 * pi * i * l / N)) for l in range(1, int(N / 2 - 1))])
    a = insert(a, 0, np.array([(1 / N) * sum(s * cos(0))]))
    a = append(a, np.array([(1 / N) * sum(s * cos(pi * i))]))

    b = np.array([(2 / N) * np.sum(s * np.sin(2 * np.pi * i * j / N)) for j in range(int(N / 2))])

    c = sqrt(pow(a, 2) + pow(a, 2))

    return a, b, c


def reverse_fourier_transform(a, b, value):
    j = arange(int(N / value))

    return np.array([sum(a[j] * cos(2 * pi * j * q / N)) + sum(b[j] * sin(2 * pi * j * q / N)) for q in range(N)])


def plotting_graph(s, a, b, c, r1):
    _, ax = plt.subplots(5, 1, figsize=(15, 10))

    ax[0].set_title("s2")
    ax[0].plot(s, color='green')

    ax[1].set_title("A")
    ax[1].plot(a, color='yellow')

    ax[2].set_title("B")
    ax[2].plot(b, color='red')

    ax[3].set_title("C")
    ax[3].plot(c, color='purple')

    ax[4].set_title("reverse Fourier transform")
    ax[4].plot(r1, color='black')

    plt.tight_layout()
    plt.show()

N=len(s2)


print("Перетворення Фур'є")
a, b, c = transformation(s2)

print("Обернене перетворення Фур'є")
r = reverse_fourier_transform(a, b, 2) 
plotting_graph(s=s2, a=a, b=b, c=c, r1=r)


va = np.array([0.42 - 0.5 * np.cos(2 * np.pi * i / N) + 0.08 * np.cos(4 * np.pi * i / N)
              for i in range(N)])

s1 = s2 * va
a, b, c = transformation(s=s1)
r = reverse_fourier_transform(a, b, 2) # Перший варіант – за всіма N/2 точками
plotting_graph(s1, a, b, c, r)

print(len(s1))

plt.title('Difference between: s1 vs r for N/2')
plt.plot(s1-r)
plt.show()



vb = np.array([0.54 - 0.46 * np.cos(2 * np.pi * i / N) for i in range(N)])

s1 = s2 * vb
a, b, c = transformation(s=s1)
r = reverse_fourier_transform(a, b, 8)
plotting_graph(s=s1, a=a, b=b, c=c, r1=r)

plt.plot(s1-r, color='cadetblue')
plt.title('Difference between: s1 vs r for N/8')
plt.show()


r = reverse_fourier_transform(a, b, 64)
plotting_graph(s=s1, a=a, b=b, c=c, r1=r)
plt.plot(s1-r, color='palevioletred')
plt.title('Difference between: s1 vs r for N/64')
plt.show()







