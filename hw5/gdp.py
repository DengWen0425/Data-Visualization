import numpy as np
import xlrd
from typing import List
import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Timeline, Grid, Bar, Map, Pie, Line


# 读取数据
gdp_file = "./data/GDP-fromworldbank.xls"
gdp = xlrd.open_workbook(gdp_file)
data = gdp.sheet_by_name("Data")
# 获取2016年度的世界GDP数据，并为他们排序，选取其中的前100个经济体
gdp2016 = data.col_values(60)[4:]
gdp2016 = np.array([x if x != "" else 0 for x in gdp2016])
top100 = gdp2016.argsort()[::-1][:100]
# 获取年份信息
label = data.row_values(3)


# --------------------------------(1.)----------------------------------------------------------
# 以下国家缩写，为从上面的2016年的 gdp 数据中手动筛选出来的前十个国家。
countries_list = ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "ITA", "BRA", "CAN"]

# 用于储存前十个国家的名字以及他们的gdp数据
countries_top10 = []
country_gdp_top10 = []

# 获取前十个国家的名字及GDP数据
for i in top100:
    tmp = data.row_values(i+4)
    if tmp[1] in countries_list:
        countries_top10.append(tmp[:2])
        country_gdp_top10.append([x/1e12 for x in tmp[41:-1]])  # 转化为万亿作为单位，并只取最近的20年，即1997~2016
# recent 20 years
year = label[41:-1]

# 自定义颜色
colors = ["#73a373", "#dd6b66", "#228d3c", "#7b6bad", "#eedd78", "#90533c", "#52b1bd", "#904790", "#aa2f2f", "#0dab00"]
# 初始化一个折线图，并定义大小和主题风格
line = Line(init_opts=opts.InitOpts(width="1200px", height="700px", theme=ThemeType.DARK))
# 添加 x 轴坐标
line.add_xaxis(xaxis_data=year)
# 逐个添加 y 轴的数据，并设置他们对应的线条颜色
for i in range(len(country_gdp_top10)):
    line.add_yaxis(
        series_name=countries_top10[i][0],
        y_axis=country_gdp_top10[i],
        label_opts=opts.LabelOpts(is_show=False, color=colors[i]),
        linestyle_opts=opts.LineStyleOpts(color=colors[i]),
        itemstyle_opts=opts.ItemStyleOpts(color=colors[i])
    )

# 折线图的全局设置，设计标题、交互式的形式、坐标轴的属性、图例的样式。
line.set_global_opts(
        title_opts=opts.TitleOpts(title="1997~2016年十个国家GDP走势图", subtitle="（2016年GDP前十的国家）"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            axislabel_opts=opts.LabelOpts(formatter="{value} 万亿")
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        legend_opts=opts.LegendOpts(type_="scroll", pos_bottom=0, orient="horizontal")
    )
# 输出并保存
line.render("./result/gdp_line.html")
# ------------------------------------------------------------------------------------------------

# --------------------------------(2.)----------------------------------------------------------
# 在这个图中，选择了2016年 gdp 数值排名前30的国家
countries_list = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "ITA", "BRA", "CAN", "KOR", "RUS", "ESP", "AUS", "MEX", "IDN",
    "TUR", "NLD", "CHE", "SAU", "ARG", "SWE", "POL", "BEL", "IRN", "THA", "NGA", "AUT", "NOR", "ARE"
]

# 用于保存适合pyecharts的数据格式，30个国家的名称，以及他们的gdp数据
gdp_data = []
countries_top30 = []
country_gdp_top30 = []
# 获取世界总量的gdp数据，30个国家的名称，以及他们的gdp数据
for i in top100:
    tmp = data.row_values(i+4)
    if tmp[1] == "WLD":
        total_num = [x/1e12 for x in tmp[41:-1]]  # 世界总量的gdp数据 单位为万亿
    if tmp[1] in countries_list:
        # 这一步是为了能够在地图上正确显示国家的区域，因此需要将他们的名字修改为地图上对应的名字
        if tmp[1] == "RUS":
            tmp[0] = "Russia"
        if tmp[1] == "IRN":
            tmp[0] = "Iran"
        if tmp[1] == "KOR":
            tmp[0] = "Korea"
        countries_top30.append(tmp[:2])
        country_gdp_top30.append([x/1e8 for x in tmp[41:-1]])  # 单位为亿

# 将30个国家的gdp数据转为 符合 pyecharts 输入的格式
for i in range(len(year)):
    tmp1 = {"time": year[i] + "年", "data": []}
    for j in range(len(countries_list)):
        tmp2 = {"name": countries_top30[j][0], "value": [country_gdp_top30[j][i], countries_top30[j][0]]}
        tmp1["data"].append(tmp2)
    gdp_data.append(tmp1)

# 修改年份的数据格式
years = [x + "年" for x in year]

# gdp数据的最大值和最小值 用于制作图例
minNum, maxNum = 320, 186245


# 定义函数制作每一年的可视化图表
def get_year_chart(year):
    # 将数据转为符合 地图输入 的形式
    map_data = [
        [[x["name"], x["value"]] for x in d["data"]] for d in gdp_data if d["time"] == year
    ][0]
    min_data, max_data = (minNum, maxNum)
    # 用于保存折线图中每年需要mark的数据点
    data_mark: List = []
    i = 0
    for x in years:
        if x == year:
            data_mark.append(total_num[i])
            total_gdp = total_num[i]
        else:
            data_mark.append("")
        i = i + 1
    # 定义一个地图 展示30个国家每年的 gdp 数值
    map_chart = (
        Map()
        .add(
            series_name="",
            data_pair=map_data,
            maptype="world",
            zoom=0.75,  # 放大倍数
            center=[119.5, 34.5],  # 地图的中心位置
            is_map_symbol_show=False,  # 不显示地图地区名
            itemstyle_opts={  # 地图区域颜色
                "normal": {"areaColor": "#323c48", "borderColor": "#404a59"},
                "emphasis": {
                    "label": {"show": Timeline},
                    "areaColor": "rgba(255,255,255, 0.5)",
                },
            },
            label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
            tooltip_opts=opts.TooltipOpts(  # 交互式触发的形式
                formatter="{b}: {c}"
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(  # 标题的设置
                title="" + str(year) + "世界30个国家年GDP情况（单位：亿）",
                subtitle="30个国家为2016年世界GDP前三十",
                pos_left="center",
                pos_top="top",
                title_textstyle_opts=opts.TextStyleOpts(
                    font_size=25, color="rgba(255,255,255, 0.9)"
                ),
            ),
            tooltip_opts=opts.TooltipOpts(  # 交互时触发的形式
                is_show=True,
                formatter=JsCode(
                    """function(params) {
                    if ('value' in params.data) {
                        return params.data.value[1] + ': ' + params.data.value[0];
                    }
                }"""
                ),
            ),
            visualmap_opts=opts.VisualMapOpts(  # 视觉映射的定义，用于制作图例
                is_calculable=True,
                dimension=0,
                pos_left="30",
                pos_top="center",
                range_text=["High", "Low"],
                range_color=["lightskyblue", "yellow", "orangered"],
                textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                min_=min_data,
                max_=max_data,
            ),
        )
    )
    # 定义一个折线图， 展示世界gdp总量的变换
    line_chart = (
        Line()
        .add_xaxis(years)
        .add_yaxis("", total_num)
        .add_yaxis(
            "",
            data_mark,
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="世界GDP总量1997-2016年（单位：万亿）", pos_left="72%", pos_top="5%"
            )
        )
    )
    # 定义一个条形图，可视化30个国家的gdp数值
    bar_x_data = [x[0] for x in map_data]
    bar_y_data = [{"name": x[0], "value": int(x[1][0]+1)} for x in map_data]
    bar = (
        Bar()
        .add_xaxis(xaxis_data=bar_x_data)
        .add_yaxis(
            series_name="",
            yaxis_data=bar_y_data,
            label_opts=opts.LabelOpts(
                is_show=True, position="right", formatter="{b} : {c}"
            ),
        )
        .reversal_axis()  # 翻转坐标轴
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                max_=maxNum, axislabel_opts=opts.LabelOpts(is_show=False)
            ),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            tooltip_opts=opts.TooltipOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(  # 视觉映射的设置
                is_calculable=True,
                dimension=0,
                pos_left="10",
                pos_top="top",
                range_text=["High", "Low"],
                range_color=["lightskyblue", "yellow", "orangered"],
                textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                min_=min_data,
                max_=max_data,
            ),
        )
    )
    # 制作一个饼图，可视化30个国家和其他国家占gdp总量的数值
    pie_data = [[x[0], x[1][0]] for x in map_data]
    others_gdp = total_gdp*1e4 - sum([x["value"] for x in bar_y_data])
    pie_data.append(["others", others_gdp])
    pie = (
        Pie()
        .add(
            series_name="",
            data_pair=pie_data,
            radius=["15%", "35%"],
            center=["80%", "82%"],
            itemstyle_opts=opts.ItemStyleOpts(
                border_width=1, border_color="rgba(0,0,0,0.3)"
            ),
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=True, formatter="{b} {d}%"),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    # 制作一个组合图表， 将上面几个可视化图放在一个图表中
    grid_chart = (
        Grid()
        .add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="10", pos_right="45%", pos_top="50%", pos_bottom="5"
            ),
        )
        .add(
            line_chart,
            grid_opts=opts.GridOpts(
                pos_left="65%", pos_right="80", pos_top="10%", pos_bottom="50%"
            ),
        )
        .add(pie, grid_opts=opts.GridOpts(pos_left="45%", pos_top="60%"))
        .add(map_chart, grid_opts=opts.GridOpts())
    )

    return grid_chart


# 制作一个 timeline 用于可视化每一年的数据
timeline = Timeline(
    init_opts=opts.InitOpts(width="1400px", height="700px", theme=ThemeType.DARK)
)
for y in years:
    g = get_year_chart(year=y)
    timeline.add(g, time_point=str(y))

timeline.add_schema(
    orient="vertical",
    is_auto_play=True,
    is_inverse=True,
    play_interval=5000,  # 自动播放的时间间隔
    pos_left="null",
    pos_right="5",
    pos_top="20",
    pos_bottom="20",
    width="60",
    label_opts=opts.LabelOpts(is_show=True, color="#fff"),
)
# 输出并保存结果
timeline.render("./result/World_gdp_1997_2016.html")





