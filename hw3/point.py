
import matplotlib.pyplot as plt
from PIL import Image


def on_press(event):
    with open("points.txt", "a") as f:
        f.write(str(event.xdata) + "\t" + str(event.ydata) + "\n")
    print("my position:",event.button,event.xdata, event.ydata)


fig = plt.figure(figsize=(25, 25))
img = Image.open('./results/freq_image.png').convert("L")

plt.imshow(img, cmap=plt.get_cmap('gray'), animated=True)

fig.canvas.mpl_connect('button_press_event', on_press)
plt.show()





