import matplotlib.pyplot as plt
import seaborn as sns
import random
# data={
#     '0':1,
#     '1':2,
#     '2':5,
#     '3':4,
#     '4':8,
#     '5':2,
#     '6':1,
#     '7':7,
#     '8':3,
#     '9':3,
# }
data={
    '0':1,
    '1':1,
    '2':1,
    '3':1,
    '4':1,
    '5':1,
    '6':1,
    '7':1,
    '8':1,
    '9':1,
}


colors= sns.color_palette('pastel')[0:20]
random.shuffle(colors)
plt.pie(list(data.values()), colors=colors)

plt.axis('equal')
# plt.show()

plt.savefig('temp_pie.png', transparent =True)
