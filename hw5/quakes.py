from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pyecharts.options as opts
from pyecharts.globals import ThemeType, GeoType
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Geo, Page

# 读取数据并存储到字典中方便后面使用。

file = open("./data/quakes.csv")
content = file.readlines()
quakes = {
    "id": [],
    "lat": [],
    "long": [],
    "depth": [],
    "mag": []
}
for i in range(1, len(content)):
    row = content[i].strip().replace('"', '').split(",")
    quakes["id"].append("id"+row[0])
    quakes["lat"].append(float(row[1]))
    quakes["long"].append(float(row[2]))
    quakes["depth"].append(float(row[3]))
    quakes["mag"].append(float(row[4]))
file.close()


# ---------------------------------- visualizations using basemap------------------------------------------
"""
以下代码为使用 Basemap 进行的可视化结果
"""
fig = plt.figure(figsize=(15, 15))
bmap = Basemap(projection='lcc', resolution=None, width=8E6, height=8E6, lat_0=-20, lon_0=180)

# map.etopo(scale=0.5, alpha=0.5)
# 绘制阴影的浮雕图像。
bmap.shadedrelief()

# 绘制经纬线
lat_lines = np.arange(-80, 80, 20.)
bmap.drawparallels(lat_lines, labels=[True, False, True, False], fontsize=20)
lon_lines = np.arange(-180., 181., 30.)
bmap.drawmeridians(lon_lines, labels=[False, False, False, True], fontsize=20)

# 转为np.array 形式 方便后面调整点的大小
quakes["mag"] = np.array(quakes["mag"])
# 在地图上绘制散点图
scatter = bmap.scatter(
    np.array(quakes["long"]), np.array(quakes["lat"]), latlon=True, s=100 * quakes["mag"]/quakes["mag"].max(),
    c=quakes["depth"], cmap=plt.get_cmap("YlOrRd"), alpha=1
)
# 可视化设置，并保存结果
plt.title("Earthquakes")
plt.colorbar(scatter, shrink=0.5)
plt.savefig("./result/earthquakes1.png", quality=100)
plt.show()

# -----------------------------------------------------------------------------------------------------------


# ---------------------------------- visualizations using pyecharts------------------------------------------
"""
以下代码为使用 Pyecharts 进行的可视化结果
"""
# 定义并初始化地理坐标系
geo1 = Geo(init_opts=opts.InitOpts(width="1200px", height="600px", theme=ThemeType.VINTAGE))
geo1.add_schema(
    maptype="world", zoom=3, center=[180, -20], label_opts=opts.LabelOpts(is_show=False),
)
geo2 = Geo(init_opts=opts.InitOpts(width="1200px", height="600px", theme=ThemeType.VINTAGE))
geo2.add_schema(
    maptype="world", zoom=3, center=[180, -20], label_opts=opts.LabelOpts(is_show=False),
)

# 处理经纬度以及深度和地震级数的数据，使其符合 pyecharts 的输入格式
mag_data = []
dep_data = []
for i in range(len(quakes["id"])):
    geo1.add_coordinate(quakes["id"][i], quakes["long"][i], quakes["lat"][i])
    geo2.add_coordinate(quakes["id"][i], quakes["long"][i], quakes["lat"][i])
    mag_data.append([quakes["id"][i], quakes["mag"][i]])
    dep_data.append([quakes["id"][i], quakes["depth"][i]])

# 在地理坐标系上绘制地震级数的可视化图
geo1.add(
    "", mag_data, type_=GeoType.SCATTER, symbol_size=5,
    tooltip_opts=opts.TooltipOpts(
        formatter=JsCode(
            """function(params) { return 'lon:' + params.value[0] + ', lat:' + params.value[1] + ', 
            mag:' + params.value[2]; } """
        ),
    ),
)
geo1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
geo1.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Earthquakes-Magnitude",
        pos_left="center",
        pos_top="top",
        title_textstyle_opts=opts.TextStyleOpts(
            font_size=25, color="black"
        ),
    ),
    visualmap_opts=opts.VisualMapOpts(
        is_calculable=True,
        dimension=2,
        pos_left="30",
        pos_top="center",
        range_text=["Magnitude"],
        range_color=["lightskyblue", "yellow", "orangered"],
        textstyle_opts=opts.TextStyleOpts(color="black"),
        min_=min(quakes["mag"]),
        max_=max(quakes["mag"]),
    ),
)

# 在地理坐标系上绘制地震深度的可视化图
geo2.add(
    "", dep_data, type_=GeoType.SCATTER, symbol_size=5,
    tooltip_opts=opts.TooltipOpts(
        formatter=JsCode(
            """function(params) { return 'lon:' + params.value[0] + ', lat:' + params.value[1] + ', 
            depth:' + params.value[2]; } """
        ),
    ),
)
geo2.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
geo2.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Earthquakes-Depth",
        pos_left="center",
        pos_top="top",
        title_textstyle_opts=opts.TextStyleOpts(
            font_size=25, color="black"
        ),
    ),
    visualmap_opts=opts.VisualMapOpts(
        is_calculable=True,
        dimension=2,
        pos_left="30",
        pos_top="center",
        range_text=["Depth"],
        range_color=["lightskyblue", "yellow", "orangered"],
        textstyle_opts=opts.TextStyleOpts(color="black"),
        min_=min(quakes["depth"]),
        max_=max(quakes["depth"]),
    ),
)

# 将上述两个图组合起来
page = Page(layout=Page.DraggablePageLayout)

page.add(
    geo1,
    geo2
)
# 输出结果
page.render("./result/earthquake2.html")
# -----------------------------------------------------------------------------------------------------------

