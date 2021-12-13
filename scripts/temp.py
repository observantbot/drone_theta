import pandas as pd
import matplotlib.pyplot as plt







df1 = pd.read_csv('csv2/game1.csv')
df2 = pd.read_csv('csv2/game2.csv')
df3 = pd.read_csv('csv2/game3.csv')
df4 = pd.read_csv('csv2/game4.csv')
df5 = pd.read_csv('csv2/game5.csv')
df6 = pd.read_csv('csv2/game6.csv')

# print(df1.head())

plt.figure
plt.title('Analysis of Learning Saturation with steps')
plt.plot(df1['model'], df1['time_score'], label = 'theta= 90')
plt.xlabel('No. of steps * 10k')
plt.ylabel('time_score')
plt.plot(df2['model'], df2['time_score'], label = 'theta= -90')
plt.plot(df3['model'], df3['time_score'], label = 'theta= 60')
plt.plot(df4['model'], df4['time_score'], label = 'theta= -60')
plt.plot(df5['model'], df5['time_score'], label = 'theta= 30')
plt.plot(df6['model'], df6['time_score'], label = 'theta= -30')
plt.legend()
# plt.savefig('figures/saturation1.png')
plt.show()

