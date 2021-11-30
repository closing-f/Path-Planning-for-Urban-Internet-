import xlrd
import xlwt
import numpy as np
import scipy.io
import random
from pathlib import Path
import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_circle(center=(3, 3), r=50):
    x = np.linspace(center[0] - r, center[0] + r, 5000)
    y1 = np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
    y2 = -np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
    plt.plot(x, y1, 'k--')
    plt.plot(x, y2, 'k--')
    x = center[0]
    y = center[1]
    # plt.plot(x, y, color="black", marker="o", markersize=4)


def plot_poi(x, y):
    plt.plot(x, y, color="black", marker="o", markersize=6, alpha=0.5)


def render(agent_pos0, agent_pos1, agent_pos2, agent_pos3, agent_pos4, agent_pos5, n,model_name):
    temp_x = agent_pos0[:, 0]
    temp_y = agent_pos0[:, 1]
    x0 = temp_x[0:256:n + 1]
    y0 = temp_y[0:256:n + 1]

    temp_x = agent_pos1[:, 0]
    temp_y = agent_pos1[:, 1]
    x1 = temp_x[0:256:n + 1]
    y1 = temp_y[0:256:n + 1]

    temp_x = agent_pos2[:, 0]
    temp_y = agent_pos2[:, 1]
    x2 = temp_x[0:256:n + 1]
    y2 = temp_y[0:256:n + 1]

    temp_x = agent_pos3[:, 0]
    temp_y = agent_pos3[:, 1]
    x3 = temp_x[0:256:n + 1]
    y3 = temp_y[0:256:n + 1]

    temp_x = agent_pos4[:, 0]
    temp_y = agent_pos4[:, 1]
    x4 = temp_x[0:256:n + 1]
    y4 = temp_y[0:256:n + 1]

    temp_x = agent_pos5[:, 0]
    temp_y = agent_pos5[:, 1]
    x5 = temp_x[0:256:n + 1]
    y5 = temp_y[0:256:n + 1]

    if 255 % (n + 1) != 0:
        x0 = np.append(x0, agent_pos0[255][0])
        y0 = np.append(y0, agent_pos0[255][1])
        x1 = np.append(x1, agent_pos1[255][0])
        y1 = np.append(y1, agent_pos1[255][1])
        x2 = np.append(x2, agent_pos2[255][0])
        y2 = np.append(y2, agent_pos2[255][1])
        x3 = np.append(x3, agent_pos3[255][0])
        y3 = np.append(y3, agent_pos3[255][1])
        x4 = np.append(x4, agent_pos4[255][0])
        y4 = np.append(y4, agent_pos4[255][1])
        x5 = np.append(x5, agent_pos5[255][0])
        y5 = np.append(y5, agent_pos5[255][1])
    myparams = {
        'axes.labelsize': '20',
        'xtick.labelsize': '18',
        'ytick.labelsize': '18',
        'lines.linewidth': 1.3,
        'legend.fontsize': '18',
        'font.family': 'Times New Roman',
        'figure.figsize': '7, 7',  # 图片尺寸
        'grid.alpha': 0.1

    }
    plt.style.use("seaborn-deep")
    pylab.rcParams.update(myparams)
    '''
    params = {
    'axes.labelsize': '35',
    'xtick.labelsize': '27',
    'ytick.labelsize': '27',
    'lines.linewidth': 2,
    'legend.fontsize': '27',
    'figure.figsize': '12, 9'  # set figure size
    }

    pylab.rcParams.update(params)  # set figure parameter
    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #set line style
    '''
    plt.plot(x0, y0, marker='^', markersize=6)
    plt.plot(x1, y1, marker='^', markersize=6)
    plt.plot(x2, y2, marker='^', markersize=6)
    plt.plot(x3, y3, marker='^', markersize=6)
    plt.plot(x4, y4, marker='^', markersize=6)
    plt.plot(x5, y5, marker='^', markersize=6)
    plt.plot(x0[0], y0[0], marker='o', markersize=8)
    plt.plot(x1[0], y1[0], marker='o', markersize=8)
    plt.plot(x2[0], y2[0], marker='o', markersize=8)
    plt.plot(x3[0], y3[0], marker='o', markersize=8)
    plt.plot(x4[0], y4[0], marker='o', markersize=8)
    plt.plot(x5[0], y5[0], marker='o', markersize=8)
    fig1 = plt.figure(1)
    ax_values = [0, 1000, 0, 1000]
    plt.axis(ax_values)
    plt.axhline()
    plt.axvline()
    # axes = plt.subplot(111)
    # axes = plt.gca()
    # axes.set_yticks([0, 50, 100, 150, 200, 250])
    # axes.set_xticks([0, 50, 100, 150, 200])
    # axes.grid(True)  # add grid

    # plt.legend(loc="lower right")  # set legend location
    plt.ylabel('y_coordinate')  # set ystick label
    plt.xlabel('x_coordinate')  # set xstck label

    # plot_circle(center=(x0[0], y0[0]), r=150)
    # plot_circle(center=(x1[0], y1[0]), r=150)
    # plot_circle(center=(x2[0], y2[0]), r=150)
    # plot_circle(center=(x3[0], y3[0]), r=150)
    # plot_circle(center=(x4[0], y4[0]), r=150)
    # plot_circle(center=(x5[0], y5[0]), r=150)
    for i in range(10):
        for j in range(10):
            x = 50 + i * 100
            y = 50 + j * 100
            plot_poi(x, y)
    # print (model_name)
    plt_path = Path('../pdf') / ('%s.pdf' % model_name)
    
    plt.savefig(plt_path, bbox_inches='tight')
    plt.show()

def test(model_name):
    # data = xlrd.open_workbook('C:/Users/86188/Desktop/test_file_MAAC/AC_agents_trace_test8.xls')

    # model_name='pre_%s_trace_%i_model%i' % ('critic', 2, 30)
    
    
    
    model_dir = Path('../excel') / ('%s.xls'% model_name)
    
    
    data = xlrd.open_workbook(model_dir)
    # 读取数据
    table = data.sheets()[0]
    # 获取第一个sheet
    nrows = table.nrows
    # 行数  256 个 数据
    # print(nrows)
    ncols = table.ncols
    # 列数  12列
    # print(ncols)

    values_x0 = table.col_values(0)
    # 第0个agent的横坐标
    values_y0 = table.col_values(1)
    # print(values_x0)
    # 第0个agent的纵坐标
    agent_pos0 = np.empty([256, 2])
    agent_pos0[:, 0] = values_x0[1:]
    agent_pos0[:, 1] = values_y0[1:]
    # for j in range(256):
    #  print(agent_pos0[j][0] ,agent_pos0[j][1])

    values_x1 = table.col_values(2)
    # 第1个agent的横坐标
    values_y1 = table.col_values(3)
    # 第1个agent的纵坐标
    agent_pos1 = np.empty([256, 2])
    agent_pos1[:, 0] = values_x1[1:]
    agent_pos1[:, 1] = values_y1[1:]
    # for j in range(256):
    #   print(agent_pos1[j][0] ,agent_pos1[j][1])

    values_x2 = table.col_values(4)
    # 第2个agent的横坐标
    values_y2 = table.col_values(5)
    # 第2个agent的纵坐标
    agent_pos2 = np.empty([256, 2])
    agent_pos2[:, 0] = values_x2[1:]
    agent_pos2[:, 1] = values_y2[1:]
    # for j in range(256):
    #   print(agent_pos2[j][0] ,agent_pos2[j][1])

    values_x3 = table.col_values(6)
    # 第3个agent的横坐标
    values_y3 = table.col_values(7)
    # 第3个agent的纵坐标
    agent_pos3 = np.empty([256, 2])
    agent_pos3[:, 0] = values_x3[1:]
    agent_pos3[:, 1] = values_y3[1:]
    # for j in range(256):
    #   print(agent_pos3[j][0] ,agent_pos3[j][1])

    values_x4 = table.col_values(8)
    # 第4个agent的横坐标
    values_y4 = table.col_values(9)
    # 第4个agent的纵坐标
    agent_pos4 = np.empty([256, 2])
    agent_pos4[:, 0] = values_x4[1:]
    agent_pos4[:, 1] = values_y4[1:]
    # for j in range(256):
    #   print(agent_pos4[j][0] ,agent_pos4[j][1])

    values_x5 = table.col_values(10)
    # 第5个agent的横坐标
    values_y5 = table.col_values(11)
    # 第5个agent的纵坐标
    agent_pos5 = np.empty([256, 2])
    agent_pos5[:, 0] = values_x5[1:]
    agent_pos5[:, 1] = values_y5[1:]
    # for j in range(256):
    #   print(agent_pos5[j][0] ,agent_pos5[j][1])

    render(agent_pos0, agent_pos1, agent_pos2,
           agent_pos3, agent_pos4, agent_pos5, 20, model_name)



if __name__ == "__main__":

    test()
